#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/mrope_kernel/mrope_kernel_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kTokenCount = 8;
constexpr int32_t kQueryHeadCount = 32;
constexpr int32_t kKeyValueHeadCount = 8;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kTemporalSection = 24;
constexpr int32_t kHeightSection = 20;
constexpr int32_t kWidthSection = 20;
constexpr float kFrequencyBase = 5000000.0f;
constexpr std::size_t kQueryElementCount = 32768U;
constexpr std::size_t kKeyValueElementCount = 8192U;

template <typename T> class NormalizeRopeCacheTest : public test::KernelTestFixture {};
using NormalizeTypes = ::testing::Types<float, Float16, BFloat16>;
TYPED_TEST_SUITE(NormalizeRopeCacheTest, NormalizeTypes);

TYPED_TEST(NormalizeRopeCacheTest, PreservesNormRotationAndCacheRounding) {
  ASSERT_TRUE(this->backend()->InitializeCompute());
  const auto convert = [](const float value) {
    if constexpr (std::is_same_v<TypeParam, float>) {
      return value;
    } else {
      return test::LowPrecision<TypeParam>(value);
    }
  };
  const auto to_float = [](const TypeParam value) {
    if constexpr (std::is_same_v<TypeParam, float>) {
      return value;
    } else {
      return test::HostFloat(value);
    }
  };
  const backend::ScalarType type = std::is_same_v<TypeParam, float> ? backend::ScalarType::kFloat32
                                   : std::is_same_v<TypeParam, Float16>
                                       ? backend::ScalarType::kFloat16
                                       : backend::ScalarType::kBFloat16;
  for (const int32_t head_dim : {72, 128, 256}) {
    constexpr int32_t kQueryHeads = 8;
    constexpr int32_t kKvHeads = 2;
    constexpr int32_t kCapacity = 4;
    std::vector<TypeParam> query(kQueryHeads * head_dim);
    std::vector<TypeParam> key(kKvHeads * head_dim);
    std::vector<TypeParam> value(key.size());
    std::vector<TypeParam> weight(head_dim);
    for (size_t i = 0; i < query.size(); ++i) {
      query[i] = convert((static_cast<int32_t>(i % 47) - 23) / 37.0F);
    }
    for (size_t i = 0; i < key.size(); ++i) {
      key[i] = convert((static_cast<int32_t>(i % 31) - 15) / 17.0F);
      value[i] = convert((static_cast<int32_t>(i % 23) - 11) / 13.0F);
    }
    for (int32_t i = 0; i < head_dim; ++i) {
      weight[i] = convert(0.9F + (i % 11) / 37.0F);
    }
    const auto q = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), query);
    const auto k = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), key);
    const auto v = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), value);
    const auto w = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), weight);
    const auto qn = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), query.size());
    const auto kn = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), key.size());
    const auto qr = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), query.size());
    const auto kr = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), key.size());
    const auto kc =
        memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), kCapacity * key.size());
    const auto vc =
        memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), kCapacity * key.size());
    const auto expected_k =
        memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), kCapacity * key.size());
    const auto expected_v =
        memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), kCapacity * key.size());
    const auto state = memory::Buffer::CreateDeviceBufferFromVector(
        this->backend(), std::vector<int32_t>{47, 13, 29, 2});
    ASSERT_NE(q, nullptr);
    ASSERT_NE(k, nullptr);
    ASSERT_NE(v, nullptr);
    ASSERT_NE(w, nullptr);
    ASSERT_NE(qn, nullptr);
    ASSERT_NE(kn, nullptr);
    ASSERT_NE(qr, nullptr);
    ASSERT_NE(kr, nullptr);
    ASSERT_NE(kc, nullptr);
    ASSERT_NE(vc, nullptr);
    ASSERT_NE(expected_k, nullptr);
    ASSERT_NE(expected_v, nullptr);
    ASSERT_NE(state, nullptr);
    ASSERT_TRUE(kc->Fill(0U));
    ASSERT_TRUE(vc->Fill(0U));
    ASSERT_TRUE(expected_k->Fill(0U));
    ASSERT_TRUE(expected_v->Fill(0U));
    const int32_t spatial_section = head_dim / 6;
    const int32_t temporal_section = head_dim / 2 - 2 * spatial_section;
    const auto normalize = [&](const memory::BufferPtr &input, const int32_t rows,
                               const memory::BufferPtr &output) {
      return this->backend()->Compute(type, backend::RmsNorm{.input = input->view(),
                                                             .weight = w->view(),
                                                             .rows = rows,
                                                             .hidden_size = head_dim,
                                                             .epsilon = 1e-6F,
                                                             .output = output->view()});
    };
    ASSERT_TRUE(normalize(q, kQueryHeads, qn));
    ASSERT_TRUE(normalize(k, kKvHeads, kn));
    ASSERT_TRUE(
        this->backend()->Compute(type, backend::MultiAxisRope{.query = qn->view(),
                                                              .key = kn->view(),
                                                              .position_ids = state->view(),
                                                              .tokens = 1,
                                                              .query_heads = kQueryHeads,
                                                              .key_value_heads = kKvHeads,
                                                              .head_dim = head_dim,
                                                              .temporal_section = temporal_section,
                                                              .height_section = spatial_section,
                                                              .width_section = spatial_section,
                                                              .frequency_base = kFrequencyBase,
                                                              .query_output = qr->view(),
                                                              .key_output = kr->view()}));
    ASSERT_TRUE(
        this->backend()->Compute(type, backend::StoreKvCache{.key = kr->view(),
                                                             .value = v->view(),
                                                             .token_count = 1,
                                                             .key_value_size = kKvHeads * head_dim,
                                                             .start_position = 2,
                                                             .cache_capacity = kCapacity,
                                                             .key_cache = expected_k->view(),
                                                             .value_cache = expected_v->view()}));
    ASSERT_TRUE(this->backend()->Compute(
        type, backend::NormalizeRopeCache{
                  .query = q->view(),
                  .key = k->view(),
                  .value = v->view(),
                  .query_weight = w->view(),
                  .key_weight = w->view(),
                  .positions = state->view(),
                  .cache_position = state->view().Slice(3U * sizeof(int32_t), sizeof(int32_t)),
                  .query_heads = kQueryHeads,
                  .key_value_heads = kKvHeads,
                  .head_dim = head_dim,
                  .cache_capacity = kCapacity,
                  .temporal_section = temporal_section,
                  .height_section = spatial_section,
                  .width_section = spatial_section,
                  .frequency_base = kFrequencyBase,
                  .epsilon = 1e-6F,
                  .query_output = q->view(),
                  .key_cache = kc->view(),
                  .value_cache = vc->view()}));
    for (const auto &buffers :
         {std::pair{q, qr}, std::pair{kc, expected_k}, std::pair{vc, expected_v}}) {
      std::vector<TypeParam> actual;
      std::vector<TypeParam> expected;
      ASSERT_TRUE(buffers.first->template CopyDataToVector<TypeParam>(&actual));
      ASSERT_TRUE(buffers.second->template CopyDataToVector<TypeParam>(&expected));
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(to_float(actual[i]), to_float(expected[i]));
      }
    }
  }
}

class MropeKernelTest : public test::KernelTestFixture {
protected:
  static int32_t PositionAxis(const int32_t pair_index) {
    if (pair_index % 3 == 1 && pair_index < 3 * kHeightSection) {
      return 1;
    }
    return pair_index % 3 == 2 && pair_index < 3 * kWidthSection ? 2 : 0;
  }

  static void RotateReference(const std::vector<float> &input,
                              const std::vector<int32_t> &position_ids, const int32_t head_count,
                              std::vector<float> *const output) {
    const std::size_t element_count = static_cast<std::size_t>(kTokenCount) * head_count * kHeadDim;
    output->assign(element_count, 0.0f);
    for (std::size_t index = 0; index < element_count; ++index) {
      const int32_t dimension = static_cast<int32_t>(index % kHeadDim);
      const int32_t token =
          static_cast<int32_t>(index / (static_cast<std::size_t>(head_count) * kHeadDim));
      const int32_t pair_index = dimension % (kHeadDim / 2);
      const int32_t axis = PositionAxis(pair_index);
      const float position =
          static_cast<float>(position_ids[static_cast<std::size_t>(axis) * kTokenCount + token]);
      const float angle =
          position * std::pow(kFrequencyBase, -2.0f * pair_index / static_cast<float>(kHeadDim));
      const int32_t rotated_dimension =
          dimension < kHeadDim / 2 ? dimension + kHeadDim / 2 : dimension - kHeadDim / 2;
      const float rotated_sign = dimension < kHeadDim / 2 ? -1.0f : 1.0f;
      const std::size_t rotated_index = index - dimension + rotated_dimension;
      (*output)[index] =
          input[index] * std::cos(angle) + rotated_sign * input[rotated_index] * std::sin(angle);
    }
  }
};

TEST_F(MropeKernelTest, RotatesQwen3VlMultimodalPositions) {
  std::vector<float> query(kQueryElementCount);
  std::vector<float> key(kKeyValueElementCount);
  std::vector<int32_t> position_ids(3U * kTokenCount);
  for (std::size_t index = 0; index < kQueryElementCount; ++index) {
    query[index] = 0.001f * static_cast<float>(index % kHeadDim) - 0.064f;
  }
  for (std::size_t index = 0; index < kKeyValueElementCount; ++index) {
    key[index] = 0.0005f * static_cast<float>(index % kHeadDim) - 0.032f;
  }
  for (int32_t token = 0; token < kTokenCount; ++token) {
    position_ids[static_cast<std::size_t>(token)] = token + 1;
    position_ids[static_cast<std::size_t>(kTokenCount + token)] = 32 + token;
    position_ids[static_cast<std::size_t>(2 * kTokenCount + token)] = 64 + token;
  }
  std::vector<float> expected_query;
  std::vector<float> expected_key;
  RotateReference(query, position_ids, kQueryHeadCount, &expected_query);
  RotateReference(key, position_ids, kKeyValueHeadCount, &expected_key);
  const auto query_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), query);
  const auto key_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), key);
  const auto position_ids_device =
      memory::Buffer::CreateDeviceBufferFromVector(backend(), position_ids);
  const auto query_output_device =
      memory::Buffer::CreateDeviceBuffer<float>(backend(), expected_query.size());
  const auto key_output_device =
      memory::Buffer::CreateDeviceBuffer<float>(backend(), expected_key.size());
  ASSERT_TRUE((runtime::MropeKernelRuntime<float>::CompileAndLaunch(
      "mrope_float", {.options = {.grid_dim = dim3(16, 1, 1),
                                  .block_dim = dim3(256, 1, 1),
                                  .shared_memory_bytes = 0},
                      .query = query_device->device_address(),
                      .key = key_device->device_address(),
                      .position_ids = position_ids_device->device_address(),
                      .tokens = kTokenCount,
                      .query_heads = kQueryHeadCount,
                      .key_value_heads = kKeyValueHeadCount,
                      .head_dim = kHeadDim,
                      .temporal_section = kTemporalSection,
                      .height_section = kHeightSection,
                      .width_section = kWidthSection,
                      .frequency_base = kFrequencyBase,
                      .query_output = query_output_device->device_address(),
                      .key_output = key_output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual_query(expected_query.size());
  std::vector<float> actual_key(expected_key.size());
  ASSERT_TRUE(query_output_device->CopyDataToVector(&actual_query));
  ASSERT_TRUE(key_output_device->CopyDataToVector(&actual_key));
  EXPECT_TRUE(test::IsClose(actual_query, expected_query, 2.0e-5f));
  EXPECT_TRUE(test::IsClose(actual_key, expected_key, 2.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels
