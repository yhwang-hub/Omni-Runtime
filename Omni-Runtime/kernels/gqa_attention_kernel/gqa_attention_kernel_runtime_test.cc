#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <array>

#include "kernels/kernel_test_utils.h"
#include "kernels/gqa_attention_kernel/gqa_attention_cuda_runtime.h"

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

// Backend parity and validation checks exercise the operator runtime dispatch.
#include "kernels/backend_test_utils.h"
#include "kernels/gqa_attention_kernel/attention_define.h"
#include "kernels/kernel_runtime.h"

#include <algorithm>
#include <limits>

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class AttentionBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(AttentionBackendTest, test::BackendTypes);

class MixedAttentionBackendTest : public test::BackendParityFixture {};

TEST(AttentionPartitionTest, PaddedStreamGroupsMatchReferenceLaunches) {
  EXPECT_EQ(runtime::GgmlAttentionStreamGroups(512, 516, 4, 48), 96);
  EXPECT_EQ(runtime::GgmlAttentionStreamGroups(512, 1028, 4, 48), 96);
  EXPECT_EQ(runtime::GgmlAttentionStreamGroups(192, 2756, 4, 48), 96);
  EXPECT_EQ(runtime::GgmlAttentionStreamGroups(10, 2766, 4, 48), 96);
  EXPECT_EQ(runtime::GgmlAttentionStreamGroups(65, 65, 1, 48), 72);
  EXPECT_EQ(runtime::GgmlAttentionStreamGroups(4, 4, 4, 48), 0);
}

TEST_F(MixedAttentionBackendTest, GgmlStreamKPreservesPartitionRounding) {
  constexpr int32_t kTokens = 512;
  constexpr int32_t kCached = 516;
  constexpr int32_t kHeads = 24;
  constexpr int32_t kKvHeads = 4;
  constexpr int32_t kHead = 256;
  constexpr int32_t kTokenTiles = kTokens / 8;
  constexpr int32_t kKeyTiles = 24;
  const int32_t groups = runtime::GgmlAttentionStreamGroups(
      kTokens, kCached, kKvHeads, runtime::CudaKernelRuntime::MultiprocessorCount());
  ASSERT_GT(groups, 0);
  const auto round = [](const float value) { return test::BackendRound<Float16>(value); };
  const auto value = [](const int32_t token) {
    const int32_t block = token / 32;
    return static_cast<float>(block % 5 - 2) * 0.1F + static_cast<float>(block) * 0.003F;
  };
  const float probability = std::exp(-3.0F * 0.6931F);
  const auto reference = [&](const int32_t token, const int32_t kv_head, const int32_t partitions) {
    const int64_t tile = kv_head * kTokenTiles + token / 8;
    constexpr int64_t kWork = kTokenTiles * kKvHeads * kKeyTiles;
    float numerator = 0.0F;
    float denominator = 0.0F;
    for (int32_t part = partitions - 1; part >= 0; --part) {
      const int64_t begin = std::max(part * kWork / partitions, tile * kKeyTiles);
      const int64_t end = std::min((part + 1) * kWork / partitions, (tile + 1) * kKeyTiles);
      if (begin >= end) {
        continue;
      }
      float partial = 0.0F;
      float sums[4] = {};
      for (int32_t first = static_cast<int32_t>((begin - tile * kKeyTiles) * 32);
           first < std::min<int32_t>((end - tile * kKeyTiles) * 32, token + 5); first += 32) {
        for (int32_t lane = 0; lane < 4; ++lane) {
          float sum = 0.0F;
          for (int32_t key = first + lane * 2; key < first + 32; key += 8) {
            sum += key < token + 5 ? probability : 0.0F;
            sum += key + 1 < token + 5 ? probability : 0.0F;
          }
          sums[lane] += sum;
        }
        for (int32_t key = first; key < first + 32; key += 16) {
          const int32_t count = std::clamp(token + 5 - key, 0, 16);
          partial =
              round(partial + static_cast<float>(count) * round(probability) * round(value(key)));
        }
      }
      numerator += partial;
      denominator += (sums[0] + sums[2]) + (sums[1] + sums[3]);
    }
    return numerator / denominator;
  };
  std::vector<float> expected(kTokens * kHeads * kHead);
  bool has_partition_rounding = false;
  for (int32_t token = 0; token < kTokens; ++token) {
    for (int32_t head = 0; head < kHeads; ++head) {
      const float result = reference(token, head / 6, groups);
      has_partition_rounding |= std::fabs(result - reference(token, head / 6, 1)) > 1.0e-6F;
      std::fill_n(expected.begin() + (token * kHeads + head) * kHead, kHead, result);
    }
  }
  ASSERT_TRUE(has_partition_rounding);
  std::vector<float> values(kCached * kKvHeads * kHead);
  for (int32_t token = 0; token < kCached; ++token) {
    std::fill_n(values.begin() + token * kKvHeads * kHead, kKvHeads * kHead, value(token));
  }
  std::array<std::vector<float>, 2U> outputs;
  for (size_t index = 0U; index < devices().size(); ++index) {
    SCOPED_TRACE(backend::BackendName(devices()[index]->kind()));
    test::BackendTensor<float> q;
    test::BackendTensor<float> k;
    test::BackendTensor<float> v;
    test::BackendTensor<float> y;
    ASSERT_TRUE(q.Initialize(devices()[index], std::vector<float>(expected.size())));
    ASSERT_TRUE(k.Initialize(devices()[index], std::vector<float>(values.size())));
    ASSERT_TRUE(v.Initialize(devices()[index], values));
    ASSERT_TRUE(y.Initialize(devices()[index], std::vector<float>(expected.size())));
    ASSERT_TRUE(devices()[index]->Compute(backend::ScalarType::kFloat32,
                                          backend::Attention{.query = q.view(),
                                                             .key_cache = k.view(),
                                                             .value_cache = v.view(),
                                                             .query_token_count = kTokens,
                                                             .query_head_count = kHeads,
                                                             .key_value_head_count = kKvHeads,
                                                             .cached_token_count = kCached,
                                                             .start_position = 4,
                                                             .head_dim = kHead,
                                                             .is_causal = true,
                                                             .output = y.view(),
                                                             .is_fp16_attention = true,
                                                             .is_ggml_softmax = true}));
    ASSERT_TRUE(y.ReadFloats(&outputs[index]));
    test::ExpectBackendClose(outputs[index], expected, {3.0e-8F, 3.0e-7F});
  }
  EXPECT_EQ(outputs[0], outputs[1]);
}

TEST_F(MixedAttentionBackendTest, GgmlSoftmaxPreservesHalfProbabilityAndFloatDenominator) {
  constexpr int32_t kHeads = 6;
  constexpr int32_t kHead = 256;
  const float probability = std::exp(-3.0F * 0.6931F);
  const float expected_value = test::BackendRound<Float16>(probability) / probability;
  for (const int32_t tokens : {1, 4, 65}) {
    const int32_t cache_tokens = tokens == 1 ? 4 : tokens;
    std::array<std::vector<float>, 2U> outputs;
    for (size_t index = 0U; index < devices().size(); ++index) {
      const auto &device = devices()[index];
      SCOPED_TRACE(backend::BackendName(device->kind()));
      SCOPED_TRACE(tokens);
      test::BackendTensor<float> q;
      test::BackendTensor<float> k;
      test::BackendTensor<float> v;
      test::BackendTensor<float> y;
      test::BackendTensor<float> partials;
      test::BackendTensor<int32_t> position;
      ASSERT_TRUE(q.Initialize(device, std::vector<float>(tokens * kHeads * kHead, 0.0F)));
      ASSERT_TRUE(k.Initialize(device, std::vector<float>(cache_tokens * kHead, 0.0F)));
      ASSERT_TRUE(v.Initialize(device, std::vector<float>(cache_tokens * kHead, 1.0F)));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(tokens * kHeads * kHead)));
      ASSERT_TRUE(partials.Initialize(device, std::vector<float>(4 * kHeads * (kHead + 2))));
      ASSERT_TRUE(position.Initialize(device, {0}));
      const backend::Attention operation{.query = q.view(),
                                         .key_cache = k.view(),
                                         .value_cache = v.view(),
                                         .query_token_count = tokens,
                                         .query_head_count = kHeads,
                                         .key_value_head_count = 1,
                                         .cached_token_count = cache_tokens,
                                         .start_position = 0,
                                         .head_dim = kHead,
                                         .is_causal = true,
                                         .output = y.view(),
                                         .decode_partials = partials.view(),
                                         .cache_position =
                                             tokens == 1 ? position.view() : backend::DeviceView{},
                                         .is_fp16_attention = true,
                                         .is_ggml_softmax = true};
      ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, operation));
      ASSERT_TRUE(y.ReadFloats(&outputs[index]));
      const std::vector<float> expected(outputs[index].size(), expected_value);
      test::ExpectBackendClose(outputs[index], expected, {3.0e-7F, 0.0F});
      const auto view = y.view();
      const backend::Allocation readback{.data = view.data,
                                         .size_bytes = view.size_bytes,
                                         .memory_kind = backend::MemoryKind::kDeviceView,
                                         .offset_bytes = view.offset_bytes};
      std::unique_ptr<backend::ExecutionPlan> plan;
      ASSERT_TRUE(
          device->CreateExecutionPlan(backend::ScalarType::kFloat32, {operation}, readback, &plan));
      double elapsed_ms = 0.0;
      ASSERT_TRUE(plan->Run(view.size_bytes, outputs[index].data(), &elapsed_ms));
      test::ExpectBackendClose(outputs[index], expected, {3.0e-7F, 0.0F});
    }
    EXPECT_EQ(outputs[0], outputs[1]);
  }
}

TEST(AttentionValidationTest, RejectsOverflowingTensorExtentsBeforeLaunch) {
  constexpr int32_t kLarge = 1 << 30;
  EXPECT_TRUE(runtime::IsValidAttentionShape(512, 32, 4, 8192, 256));
  EXPECT_FALSE(runtime::IsValidAttentionShape(kLarge, kLarge, 4, 8192, 256));
  EXPECT_FALSE(runtime::IsValidAttentionShape(1, kLarge, kLarge, kLarge, 256));
  EXPECT_FALSE(runtime::IsValidAttentionShape(1, 32, 0, 8192, 256));
  EXPECT_FALSE(runtime::IsValidAttentionShape(1, 32, 4, 8192, 0));
}

TEST_F(MixedAttentionBackendTest, PackedGgmlCachePreservesPrefillAndResidentDecodeExactly) {
  constexpr int32_t kCapacity = 577;
  constexpr int32_t kHeads = 16;
  constexpr int32_t kKvHeads = 4;
  constexpr int32_t kHead = 256;
  const int32_t partitions = runtime::CudaKernelRuntime::MultiprocessorCount() / kKvHeads;
  ASSERT_GT(partitions, 0);
  auto keys = test::BackendInput<float>(kCapacity * kKvHeads * kHead, 17);
  auto values = test::BackendInput<float>(keys.size(), 23);
  for (size_t index = 0U; index < keys.size(); ++index) {
    keys[index] *= 0.33333334F;
    values[index] += static_cast<float>(index % 7U) * 0.00013F;
  }
  for (const auto &device : devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<float> k;
    test::BackendTensor<float> v;
    test::BackendTensor<float> key_cache;
    test::BackendTensor<float> value_cache;
    test::BackendTensor<Float16> packed_key;
    test::BackendTensor<Float16> packed_value;
    test::BackendTensor<float> partials;
    ASSERT_TRUE(k.Initialize(device, keys));
    ASSERT_TRUE(v.Initialize(device, values));
    ASSERT_TRUE(key_cache.Initialize(device, std::vector<float>(keys.size())));
    ASSERT_TRUE(value_cache.Initialize(device, std::vector<float>(keys.size())));
    ASSERT_TRUE(packed_key.Initialize(device, std::vector<Float16>(keys.size())));
    ASSERT_TRUE(packed_value.Initialize(device, std::vector<Float16>(keys.size())));
    ASSERT_TRUE(partials.Initialize(device, std::vector<float>(partitions * kHeads * (kHead + 2))));
    backend::StoreKvCache store{.key = k.view(),
                                .value = v.view(),
                                .token_count = kCapacity,
                                .key_value_size = kKvHeads * kHead,
                                .start_position = 0,
                                .cache_capacity = kCapacity,
                                .key_cache = key_cache.view(),
                                .value_cache = value_cache.view(),
                                .is_fp16_cache = true};
    ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, store));
    store.key_cache = packed_key.view();
    store.value_cache = packed_value.view();
    store.is_fp16_key_value = true;
    ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, store));
    std::vector<float> expected;
    std::vector<float> actual;
    ASSERT_TRUE(key_cache.ReadFloats(&expected));
    ASSERT_TRUE(packed_key.ReadFloats(&actual));
    EXPECT_EQ(actual, expected);
    ASSERT_TRUE(value_cache.ReadFloats(&expected));
    ASSERT_TRUE(packed_value.ReadFloats(&actual));
    EXPECT_EQ(actual, expected);
    auto mismatched_store = store;
    mismatched_store.is_fp16_key_value = false;
    EXPECT_FALSE(device->Compute(backend::ScalarType::kFloat32, mismatched_store));
    store.key_cache.size_bytes -= sizeof(uint16_t);
    EXPECT_FALSE(device->Compute(backend::ScalarType::kFloat32, store));
    const auto position =
        memory::Buffer::CreateDeviceBufferFromVector(device, std::vector<int32_t>{kCapacity - 1});
    ASSERT_NE(position, nullptr);
    for (const int32_t tokens : {1, 17}) {
      SCOPED_TRACE(tokens);
      test::BackendTensor<float> q;
      test::BackendTensor<float> y;
      test::BackendTensor<float> packed_y;
      ASSERT_TRUE(q.Initialize(device, test::BackendInput<float>(tokens * kHeads * kHead, 31)));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(tokens * kHeads * kHead)));
      ASSERT_TRUE(packed_y.Initialize(device, std::vector<float>(tokens * kHeads * kHead)));
      backend::Attention operation{.query = q.view(),
                                   .key_cache = key_cache.view(),
                                   .value_cache = value_cache.view(),
                                   .query_token_count = tokens,
                                   .query_head_count = kHeads,
                                   .key_value_head_count = kKvHeads,
                                   .cached_token_count = kCapacity,
                                   .start_position = kCapacity - tokens,
                                   .head_dim = kHead,
                                   .is_causal = true,
                                   .output = y.view(),
                                   .decode_partials = partials.view(),
                                   .cache_position =
                                       tokens == 1 ? position->view() : backend::DeviceView{},
                                   .is_fp16_attention = true,
                                   .is_ggml_softmax = true};
      auto packed_operation = operation;
      packed_operation.key_cache = packed_key.view();
      packed_operation.value_cache = packed_value.view();
      packed_operation.output = packed_y.view();
      packed_operation.is_fp16_key_value = true;
      const auto view = packed_y.view();
      const backend::Allocation readback{.data = view.data,
                                         .size_bytes = view.size_bytes,
                                         .memory_kind = backend::MemoryKind::kDeviceView,
                                         .offset_bytes = view.offset_bytes};
      std::unique_ptr<backend::ExecutionPlan> plan;
      ASSERT_TRUE(device->CreateExecutionPlan(backend::ScalarType::kFloat32, {packed_operation},
                                              readback, &plan));
      const std::vector<int32_t> positions =
          tokens == 1 ? std::vector<int32_t>{kCapacity - 1, 0, 384, kCapacity - 1}
                      : std::vector<int32_t>{kCapacity - tokens};
      for (const int32_t start : positions) {
        ASSERT_TRUE(position->CopyFromHost(&start, sizeof(start)));
        ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, operation));
        ASSERT_TRUE(y.ReadFloats(&expected));
        actual.resize(expected.size());
        double elapsed_ms = 0.0;
        ASSERT_TRUE(plan->Run(view.size_bytes, actual.data(), &elapsed_ms));
        EXPECT_EQ(actual, expected);
      }
      auto mismatched_attention = packed_operation;
      mismatched_attention.is_fp16_key_value = false;
      EXPECT_FALSE(device->Compute(backend::ScalarType::kFloat32, mismatched_attention));
      packed_operation.key_cache.size_bytes -= sizeof(uint16_t);
      EXPECT_FALSE(device->Compute(backend::ScalarType::kFloat32, packed_operation));
    }
  }
}

TEST_F(MixedAttentionBackendTest, GgmlResidentDecodePreservesFourStreamsAndPartitionResets) {
  constexpr int32_t kCapacity = 4096;
  constexpr int32_t kHeads = 24;
  constexpr int32_t kKvHeads = 4;
  constexpr int32_t kHead = 256;
  const int32_t cap = runtime::CudaKernelRuntime::MultiprocessorCount() / kKvHeads;
  ASSERT_GT(cap, 0);
  const auto round = [](const float value) { return test::BackendRound<Float16>(value); };
  const auto value = [](const int32_t token) {
    const int32_t block = token / 16;
    return static_cast<float>(block % 7 - 3) * 0.1F + static_cast<float>(block % 13) * 0.003F;
  };
  const float probability = std::exp(-3.0F * 0.6931F);
  const auto reference = [&](const int32_t count) {
    const int32_t iterations = ((count + 255) / 256) * 4;
    const int32_t partitions = std::min(cap, iterations);
    float numerator = 0.0F;
    float denominator = 0.0F;
    for (int32_t part = partitions - 1; part >= 0; --part) {
      float numerators[4] = {};
      float denominators[4] = {};
      for (int32_t stream = 0; stream < 4; ++stream) {
        float sums[8] = {};
        for (int32_t first = part * iterations / partitions * 64;
             first < std::min((part + 1) * iterations / partitions * 64, count); first += 64) {
          const int32_t key = first + stream * 16;
          const int32_t valid = std::clamp(count - key, 0, 16);
          numerators[stream] =
              round(numerators[stream] +
                    static_cast<float>(valid) * round(probability) * round(value(key)));
          for (int32_t row = 0; row < 8; ++row) {
            sums[row] += (key + row < count ? probability : 0.0F) +
                         (key + row + 8 < count ? probability : 0.0F);
          }
        }
        denominators[stream] = ((sums[0] + sums[4]) + (sums[2] + sums[6])) +
                               ((sums[1] + sums[5]) + (sums[3] + sums[7]));
      }
      float partial = 0.0F;
      for (const float sum : numerators) {
        partial += sum;
      }
      numerator += partial;
      denominator += (denominators[0] + denominators[2]) + (denominators[1] + denominators[3]);
    }
    return numerator / denominator;
  };
  float serial = 0.0F;
  for (int32_t first = 0; first < 2767; first += 16) {
    serial = round(serial + static_cast<float>(std::min(16, 2767 - first)) * round(probability) *
                                round(value(first)));
  }
  EXPECT_GT(std::fabs(reference(2767) - serial / (2767.0F * probability)), 1.0e-6F);
  std::vector<float> values(kCapacity * kKvHeads * kHead);
  for (int32_t token = 0; token < kCapacity; ++token) {
    std::fill_n(values.begin() + token * kKvHeads * kHead, kKvHeads * kHead, value(token));
  }
  std::array<std::vector<float>, 2U> outputs;
  for (size_t index = 0U; index < devices().size(); ++index) {
    const auto &device = devices()[index];
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<float> q;
    test::BackendTensor<float> k;
    test::BackendTensor<float> v;
    test::BackendTensor<float> y;
    test::BackendTensor<float> partials;
    ASSERT_TRUE(q.Initialize(device, std::vector<float>(kHeads * kHead)));
    ASSERT_TRUE(k.Initialize(device, std::vector<float>(values.size())));
    ASSERT_TRUE(v.Initialize(device, values));
    ASSERT_TRUE(y.Initialize(device, std::vector<float>(kHeads * kHead)));
    ASSERT_TRUE(partials.Initialize(device, std::vector<float>(cap * kHeads * (kHead + 2))));
    const auto position =
        memory::Buffer::CreateDeviceBufferFromVector(device, std::vector<int32_t>{2766});
    ASSERT_NE(position, nullptr);
    const backend::Attention operation{.query = q.view(),
                                       .key_cache = k.view(),
                                       .value_cache = v.view(),
                                       .query_token_count = 1,
                                       .query_head_count = kHeads,
                                       .key_value_head_count = kKvHeads,
                                       .cached_token_count = kCapacity,
                                       .start_position = 0,
                                       .head_dim = kHead,
                                       .is_causal = true,
                                       .output = y.view(),
                                       .decode_partials = partials.view(),
                                       .cache_position = position->view(),
                                       .is_fp16_attention = true,
                                       .is_ggml_softmax = true};
    ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, operation));
    const auto view = y.view();
    const backend::Allocation readback{.data = view.data,
                                       .size_bytes = view.size_bytes,
                                       .memory_kind = backend::MemoryKind::kDeviceView,
                                       .offset_bytes = view.offset_bytes};
    std::unique_ptr<backend::ExecutionPlan> plan;
    ASSERT_TRUE(
        device->CreateExecutionPlan(backend::ScalarType::kFloat32, {operation}, readback, &plan));
    for (const int32_t start : {2766, 383, 0, 512, 2766}) {
      SCOPED_TRACE(start);
      ASSERT_TRUE(position->CopyFromHost(&start, sizeof(start)));
      std::vector<float> actual(kHeads * kHead);
      double elapsed_ms = 0.0;
      ASSERT_TRUE(plan->Run(view.size_bytes, actual.data(), &elapsed_ms));
      test::ExpectBackendClose(actual, std::vector<float>(actual.size(), reference(start + 1)),
                               {3.0e-8F, 3.0e-7F});
      outputs[index].insert(outputs[index].end(), actual.begin(), actual.end());
    }
  }
  EXPECT_EQ(outputs[0], outputs[1]);
}

TEST_F(MixedAttentionBackendTest, GgmlNonzeroDecodeMatchesSparseReference) {
  constexpr int32_t kTokens = 193;
  constexpr int32_t kHeads = 6;
  constexpr int32_t kKvHeads = 2;
  constexpr int32_t kHead = 256;
  std::vector<float> query(kHeads * kHead);
  std::vector<float> key(kTokens * kKvHeads * kHead);
  std::vector<float> value(key.size());
  std::vector<float> expected(query.size());
  for (int32_t head = 0; head < kHeads; ++head) {
    for (int32_t d = 0; d < kHead; ++d) {
      query[head * kHead + d] = 1.0F + float((head * 5 + d * 7) % 13) / 16.0F;
    }
  }
  for (int32_t token = 0; token < kTokens; ++token) {
    for (int32_t head = 0; head < kKvHeads; ++head) {
      key[(token * kKvHeads + head) * kHead + token % kHead] = float(token % 7 - 3) * 0.25F;
      for (int32_t d = 0; d < kHead; ++d) {
        value[(token * kKvHeads + head) * kHead + d] =
            token == d % kTokens ? float(head + 1) * 0.25F : 0.0F;
      }
    }
  }
  for (int32_t head = 0; head < kHeads; ++head) {
    std::vector<float> probabilities(kTokens);
    float denominator = 0.0F;
    for (int32_t token = 0; token < kTokens; ++token) {
      probabilities[token] =
          std::exp(query[head * kHead + token % kHead] * 0.0625F * float(token % 7 - 3) * 0.25F);
      denominator += probabilities[token];
    }
    for (int32_t d = 0; d < kHead; ++d) {
      expected[head * kHead + d] = probabilities[d % kTokens] *
                                   (float(head / (kHeads / kKvHeads) + 1) * 0.25F) / denominator;
    }
  }
  std::array<std::vector<float>, 2U> outputs;
  for (size_t index = 0U; index < devices().size(); ++index) {
    SCOPED_TRACE(backend::BackendName(devices()[index]->kind()));
    test::BackendTensor<float> q;
    test::BackendTensor<float> k;
    test::BackendTensor<float> v;
    test::BackendTensor<float> y;
    test::BackendTensor<float> partials;
    ASSERT_TRUE(q.Initialize(devices()[index], query));
    ASSERT_TRUE(k.Initialize(devices()[index], key));
    ASSERT_TRUE(v.Initialize(devices()[index], value));
    ASSERT_TRUE(y.Initialize(devices()[index], std::vector<float>(query.size())));
    ASSERT_TRUE(partials.Initialize(devices()[index], std::vector<float>(24 * kHeads * 258)));
    ASSERT_TRUE(devices()[index]->Compute(backend::ScalarType::kFloat32,
                                          backend::Attention{.query = q.view(),
                                                             .key_cache = k.view(),
                                                             .value_cache = v.view(),
                                                             .query_token_count = 1,
                                                             .query_head_count = kHeads,
                                                             .key_value_head_count = kKvHeads,
                                                             .cached_token_count = kTokens,
                                                             .start_position = kTokens - 1,
                                                             .head_dim = kHead,
                                                             .is_causal = true,
                                                             .output = y.view(),
                                                             .decode_partials = partials.view(),
                                                             .is_fp16_attention = true,
                                                             .is_ggml_softmax = true}));
    ASSERT_TRUE(y.ReadFloats(&outputs[index]));
    test::ExpectBackendClose(outputs[index], expected, {3.0e-6F, 2.0e-3F});
  }
  EXPECT_EQ(outputs[0], outputs[1]);
}

TEST_F(MixedAttentionBackendTest, Fp16ValueAccumulationMatchesRoundedReference) {
  constexpr int32_t kHead = 72;
  const auto round = [](const float value) { return test::BackendRound<Float16>(value); };
  for (const int32_t tokens : {1, 63, 128, 193}) {
    const auto query = test::BackendInput<float>(tokens * kHead, 3);
    const auto key = test::BackendInput<float>(query.size(), 7);
    const auto value = test::BackendInput<float>(query.size(), 4);
    std::vector<float> expected(query.size());
    for (int32_t token = 0; token < tokens; ++token) {
      std::array<std::array<float, kHead>, 2U> accumulators = {};
      std::array<float, 2U> maxima = {-INFINITY, -INFINITY};
      std::array<float, 2U> denominators = {};
      for (int32_t first = 0; first < tokens; first += 64) {
        const int32_t part = first / 64 % 2;
        const int32_t count = std::min(64, tokens - first);
        std::array<float, 64U> scores = {};
        float maximum = maxima[part];
        for (int32_t source = 0; source < count; ++source) {
          for (int32_t d = 0; d < kHead; d += 2) {
            const float x =
                round(round(query[token * kHead + d] / std::sqrt(float(kHead))) * 0.25F);
            const float y =
                round(round(query[token * kHead + d + 1] / std::sqrt(float(kHead))) * 0.25F);
            scores[source] += round(x * round(key[(first + source) * kHead + d])) +
                              round(y * round(key[(first + source) * kHead + d + 1]));
          }
          scores[source] *= 4.0F;
          maximum = std::max(maximum, scores[source] + 3.0F * 0.6931F);
        }
        const float rescale = std::exp(maxima[part] - maximum);
        denominators[part] *= rescale;
        for (float &sum : accumulators[part]) {
          sum = round(sum * round(rescale));
        }
        for (int32_t source = 0; source < count; ++source) {
          const float probability = std::exp(scores[source] - maximum);
          denominators[part] += probability;
          for (int32_t d = 0; d < kHead; ++d) {
            accumulators[part][d] = round(std::fma(round(value[(first + source) * kHead + d]),
                                                   round(probability), accumulators[part][d]));
          }
        }
        maxima[part] = maximum;
      }
      const float maximum = std::max(maxima[0], maxima[1]);
      const float a = std::exp(maxima[0] - maximum);
      const float b = std::exp(maxima[1] - maximum);
      const float denominator = a * denominators[0] + b * denominators[1];
      for (int32_t d = 0; d < kHead; ++d) {
        expected[token * kHead + d] =
            (a * accumulators[0][d] + b * accumulators[1][d]) / denominator;
      }
    }
    std::array<std::vector<float>, 2U> outputs;
    for (size_t index = 0U; index < devices().size(); ++index) {
      SCOPED_TRACE(backend::BackendName(devices()[index]->kind()));
      test::BackendTensor<float> q;
      test::BackendTensor<float> k;
      test::BackendTensor<float> v;
      test::BackendTensor<float> y;
      ASSERT_TRUE(q.Initialize(devices()[index], query));
      ASSERT_TRUE(k.Initialize(devices()[index], key));
      ASSERT_TRUE(v.Initialize(devices()[index], value));
      ASSERT_TRUE(y.Initialize(devices()[index], std::vector<float>(expected.size())));
      ASSERT_TRUE(devices()[index]->Compute(backend::ScalarType::kFloat32,
                                            backend::Attention{.query = q.view(),
                                                               .key_cache = k.view(),
                                                               .value_cache = v.view(),
                                                               .query_token_count = tokens,
                                                               .query_head_count = 1,
                                                               .key_value_head_count = 1,
                                                               .cached_token_count = tokens,
                                                               .start_position = 0,
                                                               .head_dim = kHead,
                                                               .is_causal = false,
                                                               .output = y.view(),
                                                               .is_fp16_accumulation = true}));
      ASSERT_TRUE(y.ReadFloats(&outputs[index]));
      test::ExpectBackendClose(outputs[index], expected, {3.0e-5F, 2.0e-3F});
      std::vector<Float16> half_key(key.size());
      std::vector<Float16> half_value(value.size());
      for (size_t element = 0U; element < key.size(); ++element) {
        half_key[element] = test::BackendValue<Float16>(key[element]);
        half_value[element] = test::BackendValue<Float16>(value[element]);
      }
      test::BackendTensor<Float16> packed_key;
      test::BackendTensor<Float16> packed_value;
      ASSERT_TRUE(packed_key.Initialize(devices()[index], half_key));
      ASSERT_TRUE(packed_value.Initialize(devices()[index], half_value));
      backend::Attention packed_args{.query = q.view(),
                                     .key_cache = packed_key.view(),
                                     .value_cache = packed_value.view(),
                                     .query_token_count = tokens,
                                     .query_head_count = 1,
                                     .key_value_head_count = 1,
                                     .cached_token_count = tokens,
                                     .start_position = 0,
                                     .head_dim = kHead,
                                     .is_causal = false,
                                     .output = y.view(),
                                     .is_fp16_accumulation = true,
                                     .is_fp16_key_value = true};
      ASSERT_TRUE(devices()[index]->Compute(backend::ScalarType::kFloat32, packed_args));
      std::vector<float> packed_output;
      ASSERT_TRUE(y.ReadFloats(&packed_output));
      EXPECT_EQ(packed_output, outputs[index]);
      half_key.insert(half_key.begin(), Float16{});
      half_value.insert(half_value.begin(), Float16{});
      test::BackendTensor<Float16> offset_key;
      test::BackendTensor<Float16> offset_value;
      ASSERT_TRUE(offset_key.Initialize(devices()[index], half_key));
      ASSERT_TRUE(offset_value.Initialize(devices()[index], half_value));
      packed_args.key_cache = offset_key.view();
      packed_args.value_cache = offset_value.view();
      packed_args.key_cache.offset_bytes += sizeof(uint16_t);
      packed_args.value_cache.offset_bytes += sizeof(uint16_t);
      packed_args.key_cache.size_bytes -= sizeof(uint16_t);
      packed_args.value_cache.size_bytes -= sizeof(uint16_t);
      ASSERT_TRUE(devices()[index]->Compute(backend::ScalarType::kFloat32, packed_args));
      ASSERT_TRUE(y.ReadFloats(&packed_output));
      EXPECT_EQ(packed_output, outputs[index]);
      packed_args.key_cache.size_bytes -= sizeof(uint16_t);
      EXPECT_FALSE(devices()[index]->Compute(backend::ScalarType::kFloat32, packed_args));
      packed_args.key_cache = packed_key.view();
      packed_args.is_fp16_accumulation = false;
      EXPECT_FALSE(devices()[index]->Compute(backend::ScalarType::kFloat32, packed_args));
    }
    EXPECT_EQ(outputs[0], outputs[1]);
  }
}

TEST_F(MixedAttentionBackendTest, Fp32StorageUsesFp16TensorAttentionIncludingHead256) {
  constexpr int32_t kTokens = 67;
  constexpr int32_t kCached = 131;
  constexpr int32_t kHeads = 4;
  constexpr int32_t kKvHeads = 2;
  for (const int32_t head_size : {72, 128, 256}) {
    const auto query = test::BackendInput<float>(kTokens * kHeads * head_size, 3);
    const auto key = test::BackendInput<float>(kCached * kKvHeads * head_size, 7);
    const auto value = test::BackendInput<float>(key.size(), 4);
    for (const bool is_causal : {false, true}) {
      std::vector<float> expected(query.size());
      for (int32_t token = 0; token < kTokens; ++token) {
        const int32_t limit = is_causal ? token + 33 : kCached;
        for (int32_t head = 0; head < kHeads; ++head) {
          const size_t q_first = (token * kHeads + head) * head_size;
          std::vector<double> scores(limit);
          double maximum = -std::numeric_limits<double>::infinity();
          for (int32_t source = 0; source < limit; ++source) {
            const size_t k_first = (source * kKvHeads + head / 2) * head_size;
            double sum = 0.0;
            for (int32_t d = 0; d < head_size; ++d) {
              sum += test::BackendRound<Float16>(query[q_first + d]) *
                     test::BackendRound<Float16>(key[k_first + d]);
            }
            scores[source] = sum / std::sqrt(head_size);
            maximum = std::max(maximum, scores[source]);
          }
          double denominator = 0.0;
          for (double &score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
          }
          for (int32_t d = 0; d < head_size; ++d) {
            double sum = 0.0;
            for (int32_t source = 0; source < limit; ++source) {
              sum += scores[source] * test::BackendRound<Float16>(
                                          value[(source * kKvHeads + head / 2) * head_size + d]);
            }
            expected[q_first + d] = static_cast<float>(sum / denominator);
          }
        }
      }
      std::array<std::vector<float>, 2U> outputs;
      for (size_t device_index = 0U; device_index < devices().size(); ++device_index) {
        const auto &device = devices()[device_index];
        test::BackendTensor<float> q;
        test::BackendTensor<float> k;
        test::BackendTensor<float> v;
        test::BackendTensor<float> y;
        ASSERT_TRUE(q.Initialize(device, query));
        ASSERT_TRUE(k.Initialize(device, key));
        ASSERT_TRUE(v.Initialize(device, value));
        ASSERT_TRUE(y.Initialize(device, std::vector<float>(query.size())));
        ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                    backend::Attention{.query = q.view(),
                                                       .key_cache = k.view(),
                                                       .value_cache = v.view(),
                                                       .query_token_count = kTokens,
                                                       .query_head_count = kHeads,
                                                       .key_value_head_count = kKvHeads,
                                                       .cached_token_count = kCached,
                                                       .start_position = 32,
                                                       .head_dim = head_size,
                                                       .is_causal = is_causal,
                                                       .output = y.view(),
                                                       .is_fp16_attention = true}));
        ASSERT_TRUE(y.ReadFloats(&outputs[device_index]));
        test::ExpectBackendClose(outputs[device_index], expected, {2.0e-4F, 2.0e-4F});
      }
      test::ExpectBackendClose(outputs[0], outputs[1], {2.0e-5F, 2.0e-4F});
    }
  }
}

TYPED_TEST(AttentionBackendTest, GroupedCausalPrefillAndResidentDecodeMatchReference) {
  for (const auto shape : {std::array<int32_t, 2U>{4, 72}, std::array<int32_t, 2U>{8, 128}}) {
    const int32_t kQueryHeads = shape[0];
    constexpr int32_t kKvHeads = 2;
    const int32_t kHead = shape[1];
    for (const int32_t tokens : {1, 7}) {
      const int32_t cached = tokens == 1 ? 257 : 9;
      const int32_t start = tokens == 1 ? cached - 1 : 2;
      const auto query = test::BackendInput<TypeParam>(tokens * kQueryHeads * kHead, 2);
      const auto key = test::BackendInput<TypeParam>(cached * kKvHeads * kHead, 4);
      const auto value = test::BackendInput<TypeParam>(key.size(), 9);
      std::vector<float> expected(query.size(), 0.0F);
      for (int32_t token = 0; token < tokens; ++token) {
        const int32_t limit = std::min(cached, start + token + 1);
        for (int32_t head = 0; head < kQueryHeads; ++head) {
          std::vector<double> scores(limit);
          double maximum = -std::numeric_limits<double>::infinity();
          const int32_t q = (token * kQueryHeads + head) * kHead;
          for (int32_t source = 0; source < limit; ++source) {
            const int32_t k = (source * kKvHeads + head / (kQueryHeads / kKvHeads)) * kHead;
            double sum = 0.0;
            for (int32_t d = 0; d < kHead; ++d) {
              sum += test::BackendFloat(query[q + d]) * test::BackendFloat(key[k + d]);
            }
            scores[source] = sum / std::sqrt(kHead);
            maximum = std::max(maximum, scores[source]);
          }
          double denominator = 0.0;
          for (double &score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
          }
          for (int32_t d = 0; d < kHead; ++d) {
            double sum = 0.0;
            for (int32_t source = 0; source < limit; ++source) {
              sum += scores[source] / denominator *
                     test::BackendFloat(
                         value[(source * kKvHeads + head / (kQueryHeads / kKvHeads)) * kHead + d]);
            }
            expected[q + d] = test::BackendRound<TypeParam>(sum);
          }
        }
      }
      std::array<std::vector<float>, 2U> outputs;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        test::BackendTensor<TypeParam> q;
        test::BackendTensor<TypeParam> k;
        test::BackendTensor<TypeParam> v;
        test::BackendTensor<TypeParam> y;
        test::BackendTensor<int32_t> position;
        test::BackendTensor<float> partials;
        ASSERT_TRUE(q.Initialize(this->devices()[device], query));
        ASSERT_TRUE(k.Initialize(this->devices()[device], key));
        ASSERT_TRUE(v.Initialize(this->devices()[device], value));
        ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(query.size())));
        ASSERT_TRUE(position.Initialize(this->devices()[device], {start}));
        ASSERT_TRUE(partials.Initialize(this->devices()[device],
                                        std::vector<float>(3U * kQueryHeads * (kHead + 2U))));
        ASSERT_TRUE(this->devices()[device]->Compute(
            test::BackendType<TypeParam>(),
            backend::Attention{
                .query = q.view(),
                .key_cache = k.view(),
                .value_cache = v.view(),
                .query_token_count = tokens,
                .query_head_count = kQueryHeads,
                .key_value_head_count = kKvHeads,
                .cached_token_count = cached,
                .start_position = start,
                .head_dim = kHead,
                .is_causal = true,
                .output = y.view(),
                .decode_partials = tokens == 1 ? partials.view() : backend::DeviceView{},
                .cache_position = tokens == 1 ? position.view() : backend::DeviceView{}}));
        ASSERT_TRUE(y.ReadFloats(&outputs[device]));
        test::ExpectBackendClose(outputs[device], expected,
                                 test::BackendMathTolerance<TypeParam>());
      }
      test::ExpectBackendClose(outputs[0], outputs[1], test::BackendMathTolerance<TypeParam>());
    }
  }
}

TYPED_TEST(AttentionBackendTest, PreparedDecodeCrossesPartitionsAndResetsResidentState) {
  constexpr int32_t kCapacity = 385;
  constexpr int32_t kQueryHeads = 8;
  constexpr int32_t kKvHeads = 2;
  constexpr int32_t kHead = 128;
  constexpr size_t kOutputCount = kQueryHeads * kHead;
  const auto query = test::BackendInput<TypeParam>(kOutputCount, 3);
  const auto key = test::BackendInput<TypeParam>(kCapacity * kKvHeads * kHead, 5);
  const auto value = test::BackendInput<TypeParam>(key.size(), 8);
  for (const auto &device : this->devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<TypeParam> q;
    test::BackendTensor<TypeParam> k;
    test::BackendTensor<TypeParam> v;
    test::BackendTensor<TypeParam> y;
    test::BackendTensor<TypeParam> reference;
    test::BackendTensor<float> partials;
    ASSERT_TRUE(q.Initialize(device, query));
    ASSERT_TRUE(k.Initialize(device, key));
    ASSERT_TRUE(v.Initialize(device, value));
    ASSERT_TRUE(y.Initialize(device, std::vector<TypeParam>(kOutputCount)));
    ASSERT_TRUE(reference.Initialize(device, std::vector<TypeParam>(kOutputCount)));
    ASSERT_TRUE(
        partials.Initialize(device, std::vector<float>(4U * kQueryHeads * (kHead + 2U), 97.0F)));
    const auto state =
        memory::Buffer::CreateDeviceBufferFromVector(device, std::vector<int32_t>(4U, 0));
    ASSERT_NE(state, nullptr);
    backend::Attention operation{.query = q.view(),
                                 .key_cache = k.view(),
                                 .value_cache = v.view(),
                                 .query_token_count = 1,
                                 .query_head_count = kQueryHeads,
                                 .key_value_head_count = kKvHeads,
                                 .cached_token_count = kCapacity,
                                 .start_position = 0,
                                 .head_dim = kHead,
                                 .is_causal = true,
                                 .output = y.view(),
                                 .decode_partials = partials.view(),
                                 .cache_position =
                                     state->view().Slice(3U * sizeof(int32_t), sizeof(int32_t))};
    const backend::DeviceView output = y.view();
    const backend::Allocation readback{.data = output.data,
                                       .size_bytes = output.size_bytes,
                                       .memory_kind = backend::MemoryKind::kDeviceView,
                                       .offset_bytes = output.offset_bytes};
    const std::vector<backend::ComputeOperation> operations = {
        operation,
        backend::AdvanceSequence{.positions = state->view().Slice(0U, 3U * sizeof(int32_t)),
                                 .cache_position = operation.cache_position}};
    std::unique_ptr<backend::ExecutionPlan> plan;
    ASSERT_TRUE(
        device->CreateExecutionPlan(test::BackendType<TypeParam>(), operations, readback, &plan));
    ASSERT_NE(plan, nullptr);
    std::vector<int32_t> actual_state;
    ASSERT_TRUE(state->CopyDataToVector(&actual_state));
    EXPECT_EQ(actual_state, std::vector<int32_t>(4U, 0));
    for (const int32_t start : {126, 254, 383, 0}) {
      SCOPED_TRACE(start);
      const std::vector<int32_t> reset_state(4U, start);
      ASSERT_TRUE(state->CopyFromHost(reset_state.data(), reset_state.size() * sizeof(int32_t)));
      const int32_t steps = std::min(3, kCapacity - start);
      for (int32_t step = 0; step < steps; ++step) {
        const int32_t position = start + step;
        std::vector<TypeParam> result(kOutputCount);
        double elapsed_ms = -1.0;
        ASSERT_TRUE(plan->Run(result.size() * sizeof(TypeParam), result.data(), &elapsed_ms));
        EXPECT_GT(elapsed_ms, 0.0);
        backend::Attention direct = operation;
        direct.output = reference.view();
        direct.decode_partials = {};
        direct.cache_position = {};
        direct.cached_token_count = position + 1;
        direct.start_position = position;
        ASSERT_TRUE(device->Compute(test::BackendType<TypeParam>(), direct));
        std::vector<float> expected;
        ASSERT_TRUE(reference.ReadFloats(&expected));
        test::ExpectBackendClose(test::BackendFloats(result), expected,
                                 test::BackendMathTolerance<TypeParam>());
        ASSERT_TRUE(state->CopyDataToVector(&actual_state));
        EXPECT_EQ(actual_state, std::vector<int32_t>(4U, position + 1));
      }
    }
    std::vector<TypeParam> guarded_output;
    ASSERT_TRUE(y.Read(&guarded_output));
    std::vector<float> guarded_partials;
    ASSERT_TRUE(partials.Read(&guarded_partials));
  }
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
