#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

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

class MropeKernelTest : public test::KernelTestFixture {
protected:
  static int32_t PositionAxis(const int32_t pair_index) {
    if (pair_index < kTemporalSection) {
      return 0;
    }
    return pair_index < kTemporalSection + kHeightSection ? 1 : 2;
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
  ASSERT_TRUE(Compile("mrope_float", test::MakeMropeFloatSource()));
  ASSERT_TRUE(Launch(dim3(16, 1, 1), dim3(256, 1, 1), 0, query_device->device_address(),
                     key_device->device_address(), position_ids_device->device_address(),
                     kTokenCount, kQueryHeadCount, kKeyValueHeadCount, kHeadDim, kTemporalSection,
                     kHeightSection, kWidthSection, kFrequencyBase,
                     query_output_device->device_address(), key_output_device->device_address()));
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
