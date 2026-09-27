#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kTokenCount = 8;
constexpr int32_t kHiddenSize = 4096;
constexpr float kEpsilon = 1.0e-6f;
constexpr std::size_t kElementCount = 32768U;

class LayernormKernelTest : public test::KernelTestFixture {};

TEST_F(LayernormKernelTest, NormalizesQwen3VlHiddenState) {
  std::vector<float> input(kElementCount);
  std::vector<float> weight(kHiddenSize);
  std::vector<float> bias(kHiddenSize);
  std::vector<float> expected(kElementCount);
  for (int32_t row = 0; row < kTokenCount; ++row) {
    float mean = 0.0f;
    float square_mean = 0.0f;
    for (int32_t dimension = 0; dimension < kHiddenSize; ++dimension) {
      const int64_t index = static_cast<int64_t>(row) * kHiddenSize + dimension;
      const float value = 0.001f * static_cast<float>(dimension) - 2.0f;
      input[index] = value;
      mean += value;
      square_mean += value * value;
    }
    mean /= static_cast<float>(kHiddenSize);
    square_mean /= static_cast<float>(kHiddenSize);
    const float inverse_std =
        1.0f / std::sqrt(std::max(square_mean - mean * mean, 0.0f) + kEpsilon);
    for (int32_t dimension = 0; dimension < kHiddenSize; ++dimension) {
      const int64_t index = static_cast<int64_t>(row) * kHiddenSize + dimension;
      weight[dimension] = 1.0f + 0.0001f * static_cast<float>(dimension);
      bias[dimension] = 0.001f * static_cast<float>(dimension);
      expected[index] = (input[index] - mean) * inverse_std * weight[dimension] + bias[dimension];
    }
  }
  const auto input_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto weight_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), weight);
  const auto bias_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), bias);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("layernorm_float", test::MakeLayernormFloatSource()));
  ASSERT_TRUE(Launch(dim3(kTokenCount, 1, 1), dim3(256, 1, 1), 0, input_device->device_address(),
                     weight_device->device_address(), bias_device->device_address(), kTokenCount,
                     kHiddenSize, kEpsilon, output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-4f));
}
} // namespace
} // namespace omni_runtime::kernels
