#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <array>

#include "kernels/kernel_test_utils.h"
#include "kernels/gqa_attention_kernel/gqa_attention_kernel_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kQueryTokenCount = 4;
constexpr int32_t kQueryHeadCount = 32;
constexpr int32_t kKeyValueHeadCount = 8;
constexpr int32_t kCachedTokenCount = 8;
constexpr int32_t kStartPosition = 4;
constexpr int32_t kHeadDim = 128;
constexpr std::size_t kQueryElementCount = 16384U;
constexpr std::size_t kKeyValueElementCount = 8192U;

class GqaAttentionKernelTest : public test::KernelTestFixture {};

template <typename T> class GqaAttentionLowPrecisionTest : public test::KernelTestFixture {};
using AttentionTypes = ::testing::Types<Float16, BFloat16>;
TYPED_TEST_SUITE(GqaAttentionLowPrecisionTest, AttentionTypes);

TYPED_TEST(GqaAttentionLowPrecisionTest, ReplaysResidentLengthsAcrossPartitionsAndResets) {
  ASSERT_TRUE(this->backend()->InitializeCompute());
  constexpr int32_t kCapacity = 385;
  constexpr int32_t kKvHeads = 2;
  for (const int32_t groups : {2, 4}) {
    for (const int32_t head_dim : {72, 128}) {
      const int32_t heads = kKvHeads * groups;
      std::vector<TypeParam> query(heads * head_dim);
      std::vector<TypeParam> key(kCapacity * kKvHeads * head_dim);
      std::vector<TypeParam> value(key.size());
      for (size_t i = 0; i < query.size(); ++i) {
        query[i] = test::LowPrecision<TypeParam>((static_cast<int32_t>(i % 37) - 18) / 17.0F);
      }
      for (size_t i = 0; i < key.size(); ++i) {
        key[i] = test::LowPrecision<TypeParam>((static_cast<int32_t>(i % 43) - 21) / 23.0F);
        value[i] = test::LowPrecision<TypeParam>((static_cast<int32_t>(i % 47) - 23) / 29.0F);
      }
      const auto q = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), query);
      const auto k = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), key);
      const auto v = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), value);
      const auto output =
          memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), query.size());
      const auto reference =
          memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), query.size());
      const auto partials =
          memory::Buffer::CreateDeviceBuffer<float>(this->backend(), 4U * heads * (head_dim + 2));
      const auto state = memory::Buffer::CreateDeviceBuffer<int32_t>(this->backend(), 4U);
      ASSERT_NE(q, nullptr);
      ASSERT_NE(k, nullptr);
      ASSERT_NE(v, nullptr);
      ASSERT_NE(output, nullptr);
      ASSERT_NE(reference, nullptr);
      ASSERT_NE(partials, nullptr);
      ASSERT_NE(state, nullptr);
      const backend::DeviceView cache_position =
          state->view().Slice(3U * sizeof(int32_t), sizeof(int32_t));
      const backend::ScalarType type = std::is_same_v<TypeParam, Float16>
                                           ? backend::ScalarType::kFloat16
                                           : backend::ScalarType::kBFloat16;
      const std::vector<backend::ComputeOperation> operations = {
          backend::Attention{.query = q->view(),
                             .key_cache = k->view(),
                             .value_cache = v->view(),
                             .query_token_count = 1,
                             .query_head_count = heads,
                             .key_value_head_count = kKvHeads,
                             .cached_token_count = kCapacity,
                             .start_position = 0,
                             .head_dim = head_dim,
                             .is_causal = true,
                             .output = output->view(),
                             .decode_partials = partials->view(),
                             .cache_position = cache_position},
          backend::AdvanceSequence{.positions = state->view(), .cache_position = cache_position}};
      std::unique_ptr<backend::ExecutionPlan> plan;
      ASSERT_TRUE(
          this->backend()->CreateExecutionPlan(type, operations, output->allocation(), &plan));
      for (const int32_t start : {126, 254, 383, 0}) {
        const std::vector<int32_t> initial_state = {7, 7, 7, start};
        ASSERT_TRUE(state->CopyFromHost(initial_state.data(), 4U * sizeof(int32_t)));
        const int32_t step_count = start == 383 ? 2 : 3;
        for (int32_t step = 0; step < step_count; ++step) {
          const int32_t position = start + step;
          ASSERT_TRUE((runtime::GqaAttentionKernelRuntime<TypeParam>::CompileAndLaunch(
              "decode_length_reference", {.options = {},
                                          .query = q->device_address(),
                                          .key_cache = k->device_address(),
                                          .value_cache = v->device_address(),
                                          .query_token_count = 1,
                                          .query_head_count = heads,
                                          .key_value_head_count = kKvHeads,
                                          .cached_token_count = position + 1,
                                          .start_position = position,
                                          .head_dim = head_dim,
                                          .is_causal = true,
                                          .output = reference->device_address(),
                                          .decode_partials = partials->device_address()})));
          std::vector<TypeParam> expected;
          ASSERT_TRUE(reference->template CopyDataToVector<TypeParam>(&expected));
          ASSERT_TRUE(partials->Fill(0xff));
          std::vector<TypeParam> actual(query.size());
          double elapsed_ms = 0.0;
          ASSERT_TRUE(plan->Run(actual.size() * sizeof(TypeParam), actual.data(), &elapsed_ms));
          EXPECT_GT(elapsed_ms, 0.0);
          for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_EQ(actual[i].bits, expected[i].bits);
          }
          std::vector<int32_t> updated_state;
          ASSERT_TRUE(state->template CopyDataToVector<int32_t>(&updated_state));
          EXPECT_EQ(updated_state,
                    (std::vector<int32_t>{8 + step, 8 + step, 8 + step, position + 1}));
        }
      }
    }
  }
}

TYPED_TEST(GqaAttentionLowPrecisionTest, MatchesFloatForTailTilesGroupedHeadsAndSplitDecode) {
  const std::array<std::array<int32_t, 4U>, 10U> cases = {{{67, 67, 72, 0},
                                                           {67, 137, 128, 70},
                                                           {1, 257, 128, 256},
                                                           {1, 257, 72, 1},
                                                           {1, 1, 128, 0},
                                                           {1, 129, 128, 0},
                                                           {1, 257, 128, 1},
                                                           {1, 127, 128, 126},
                                                           {1, 128, 128, 127},
                                                           {3, 133, 77, 1}}};
  for (const int32_t groups : {2, 4}) {
    for (const auto &dimensions : cases) {
      for (const int32_t offset : {0, 1}) {
        SCOPED_TRACE(::testing::Message()
                     << "groups=" << groups << ", tokens=" << dimensions[0]
                     << ", cached=" << dimensions[1] << ", dim=" << dimensions[2]
                     << ", start=" << dimensions[3] << ", offset=" << offset);
        const int32_t tokens = dimensions[0];
        const int32_t cached = dimensions[1];
        const int32_t head_dim = dimensions[2];
        const int32_t kHeads = 2 * groups;
        constexpr int32_t kKvHeads = 2;
        const size_t query_count = tokens * kHeads * head_dim;
        const size_t kv_count = cached * kKvHeads * head_dim;
        std::vector<float> query(query_count);
        std::vector<float> key(kv_count);
        std::vector<float> value(kv_count);
        std::vector<TypeParam> typed_query(query_count + offset);
        std::vector<TypeParam> typed_key(kv_count + offset);
        std::vector<TypeParam> typed_value(kv_count + offset);
        for (size_t index = 0U; index < query_count; ++index) {
          query[index] = (static_cast<int32_t>(index % 17U) - 8) / 64.0F;
          if (cached == 129) {
            query[index] *= 64.0F;
          }
          typed_query[index + offset] = test::LowPrecision<TypeParam>(query[index]);
        }
        for (size_t index = 0U; index < kv_count; ++index) {
          key[index] = (static_cast<int32_t>(index % 13U) - 6) / 64.0F;
          value[index] = (static_cast<int32_t>(index % 11U) - 5) / 32.0F;
          typed_key[index + offset] = test::LowPrecision<TypeParam>(key[index]);
          typed_value[index + offset] = test::LowPrecision<TypeParam>(value[index]);
        }
        const auto q = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), query);
        const auto k = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), key);
        const auto v = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), value);
        const auto tq = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), typed_query);
        const auto tk = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), typed_key);
        const auto tv = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), typed_value);
        const auto reference =
            memory::Buffer::CreateDeviceBuffer<float>(this->backend(), query_count);
        const auto output =
            memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), query_count);
        const auto partials = memory::Buffer::CreateDeviceBuffer<float>(
            this->backend(), ((cached + 127) / 128) * kHeads * (head_dim + 2));
        ASSERT_TRUE((runtime::GqaAttentionKernelRuntime<float>::CompileAndLaunch(
            "attention_float_reference",
            {.options = {.grid_dim = {static_cast<uint32_t>(tokens * kHeads), 1U, 1U},
                         .block_dim = {128U, 1U, 1U},
                         .shared_memory_bytes = 0},
             .query = q->device_address(),
             .key_cache = k->device_address(),
             .value_cache = v->device_address(),
             .query_token_count = tokens,
             .query_head_count = kHeads,
             .key_value_head_count = kKvHeads,
             .cached_token_count = cached,
             .start_position = dimensions[3],
             .head_dim = head_dim,
             .is_causal = dimensions[3] > 0,
             .output = reference->device_address()})));
        ASSERT_TRUE((runtime::GqaAttentionKernelRuntime<TypeParam>::CompileAndLaunch(
            "attention_low_precision",
            {.options = {},
             .query = static_cast<const TypeParam *>(tq->device_address()) + offset,
             .key_cache = static_cast<const TypeParam *>(tk->device_address()) + offset,
             .value_cache = static_cast<const TypeParam *>(tv->device_address()) + offset,
             .query_token_count = tokens,
             .query_head_count = kHeads,
             .key_value_head_count = kKvHeads,
             .cached_token_count = cached,
             .start_position = dimensions[3],
             .head_dim = head_dim,
             .is_causal = dimensions[3] > 0,
             .output = output->device_address(),
             .decode_partials = partials->device_address()})));
        std::vector<float> expected(query_count);
        std::vector<TypeParam> actual(query_count);
        ASSERT_TRUE(reference->CopyToHost(query_count * sizeof(float), expected.data()));
        ASSERT_TRUE(output->CopyToHost(query_count * sizeof(TypeParam), actual.data()));
        for (size_t index = 0U; index < query_count; ++index) {
          ASSERT_TRUE(std::isfinite(test::HostFloat(actual[index])));
          EXPECT_NEAR(test::HostFloat(actual[index]), expected[index], 1.0e-3F);
        }
      }
    }
  }
}

TEST_F(GqaAttentionKernelTest, AttendsQwen3VlGroupedHeads) {
  std::vector<float> query(kQueryElementCount);
  std::vector<float> key(kKeyValueElementCount);
  std::vector<float> value(kKeyValueElementCount);
  for (std::size_t index = 0; index < kQueryElementCount; ++index) {
    query[index] = 0.001f * static_cast<float>(index % kHeadDim) - 0.064f;
  }
  for (std::size_t index = 0; index < kKeyValueElementCount; ++index) {
    key[index] = 0.0005f * static_cast<float>(index % kHeadDim) - 0.032f;
    value[index] = 0.0002f * static_cast<float>(index % kHeadDim) + 0.016f;
  }
  const int32_t group_size = kQueryHeadCount / kKeyValueHeadCount;
  std::vector<float> expected(kQueryElementCount);
  for (int32_t query_row = 0; query_row < kQueryTokenCount * kQueryHeadCount; ++query_row) {
    const int32_t query_token = query_row / kQueryHeadCount;
    const int32_t query_head = query_row % kQueryHeadCount;
    const int32_t key_value_head = query_head / group_size;
    const int32_t source_limit = kStartPosition + query_token + 1;
    std::vector<float> scores(source_limit);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    for (int32_t source_token = 0; source_token < source_limit; ++source_token) {
      const int64_t key_offset =
          (static_cast<int64_t>(source_token) * kKeyValueHeadCount + key_value_head) * kHeadDim;
      float score = 0.0f;
      for (int32_t dimension = 0; dimension < kHeadDim; ++dimension) {
        score += query[static_cast<int64_t>(query_row) * kHeadDim + dimension] *
                 key[key_offset + dimension];
      }
      scores[source_token] = score * scale;
    }
    const float maximum = *std::max_element(scores.begin(), scores.end());
    float denominator = 0.0f;
    for (float &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (int32_t source_token = 0; source_token < source_limit; ++source_token) {
      const int64_t value_offset =
          (static_cast<int64_t>(source_token) * kKeyValueHeadCount + key_value_head) * kHeadDim;
      for (int32_t dimension = 0; dimension < kHeadDim; ++dimension) {
        expected[static_cast<int64_t>(query_row) * kHeadDim + dimension] +=
            scores[source_token] / denominator * value[value_offset + dimension];
      }
    }
  }
  const auto query_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), query);
  const auto key_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), key);
  const auto value_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), value);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE((runtime::GqaAttentionKernelRuntime<float>::CompileAndLaunch(
      "gqa_attention_float",
      {.options = {.grid_dim = dim3(kQueryTokenCount * kQueryHeadCount, 1, 1),
                   .block_dim = dim3(128, 1, 1),
                   .shared_memory_bytes = 0},
       .query = query_device->device_address(),
       .key_cache = key_device->device_address(),
       .value_cache = value_device->device_address(),
       .query_token_count = kQueryTokenCount,
       .query_head_count = kQueryHeadCount,
       .key_value_head_count = kKeyValueHeadCount,
       .cached_token_count = kCachedTokenCount,
       .start_position = kStartPosition,
       .head_dim = kHeadDim,
       .is_causal = true,
       .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 3.0e-5f));
}
TEST_F(GqaAttentionKernelTest, AttendsQwen3VlVisionHeadDim72) {
  constexpr int32_t kVisionTokenCount = 4;
  constexpr int32_t kVisionHeadCount = 2;
  constexpr int32_t kVisionHeadDim = 72;
  constexpr std::size_t kVisionElementCount = 576U;
  std::vector<float> query(kVisionElementCount);
  std::vector<float> key(kVisionElementCount);
  std::vector<float> value(kVisionElementCount);
  std::vector<float> expected(kVisionElementCount, 0.0F);
  for (std::size_t index = 0U; index < kVisionElementCount; ++index) {
    query[index] = 0.001F * static_cast<float>(index % kVisionHeadDim) - 0.04F;
    key[index] = 0.0007F * static_cast<float>(index % kVisionHeadDim) - 0.02F;
    value[index] = 0.0004F * static_cast<float>(index % kVisionHeadDim) + 0.01F;
  }
  const float scale = 1.0F / std::sqrt(static_cast<float>(kVisionHeadDim));
  for (int32_t query_row = 0; query_row < kVisionTokenCount * kVisionHeadCount; ++query_row) {
    std::vector<float> scores(kVisionTokenCount);
    for (int32_t source_token = 0; source_token < kVisionTokenCount; ++source_token) {
      float score = 0.0F;
      for (int32_t dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        score += query[static_cast<std::size_t>(query_row) * kVisionHeadDim + dimension] *
                 key[static_cast<std::size_t>(source_token) * kVisionHeadCount * kVisionHeadDim +
                     static_cast<std::size_t>(query_row % kVisionHeadCount) * kVisionHeadDim +
                     dimension];
      }
      scores[source_token] = score * scale;
    }
    const float maximum = *std::max_element(scores.begin(), scores.end());
    float denominator = 0.0F;
    for (float &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (int32_t source_token = 0; source_token < kVisionTokenCount; ++source_token) {
      const std::size_t value_offset =
          static_cast<std::size_t>(source_token) * kVisionHeadCount * kVisionHeadDim +
          static_cast<std::size_t>(query_row % kVisionHeadCount) * kVisionHeadDim;
      for (int32_t dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        expected[static_cast<std::size_t>(query_row) * kVisionHeadDim + dimension] +=
            scores[source_token] / denominator * value[value_offset + dimension];
      }
    }
  }
  const auto query_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), query);
  const auto key_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), key);
  const auto value_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), value);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE((runtime::GqaAttentionKernelRuntime<float>::CompileAndLaunch(
      "gqa_attention_vision_float",
      {.options = {.grid_dim = dim3(kVisionTokenCount * kVisionHeadCount, 1U, 1U),
                   .block_dim = dim3(128U, 1U, 1U),
                   .shared_memory_bytes = 0},
       .query = query_device->device_address(),
       .key_cache = key_device->device_address(),
       .value_cache = value_device->device_address(),
       .query_token_count = kVisionTokenCount,
       .query_head_count = kVisionHeadCount,
       .key_value_head_count = kVisionHeadCount,
       .cached_token_count = kVisionTokenCount,
       .start_position = 0,
       .head_dim = kVisionHeadDim,
       .is_causal = false,
       .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 3.0e-5F));
}
} // namespace
} // namespace omni_runtime::kernels
