#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kRowCount = 8;
constexpr int32_t kHiddenSize = 4096;
constexpr float kEpsilon = 1.0e-6f;
constexpr std::size_t kElementCount = 32768U;

class AdarmsKernelTest : public test::KernelTestFixture {};

TEST_F(AdarmsKernelTest, NormalizesQwen3VlAdaptiveHiddenState) {
  std::vector<float> input(kElementCount);
  std::vector<float> scale(kHiddenSize);
  std::vector<float> shift(kHiddenSize);
  std::vector<float> expected(kElementCount);
  for (int32_t row = 0; row < kRowCount; ++row) {
    float square_mean = 0.0f;
    for (int32_t dimension = 0; dimension < kHiddenSize; ++dimension) {
      const int64_t index = static_cast<int64_t>(row) * kHiddenSize + dimension;
      const float value = 0.002f * static_cast<float>(dimension) - 4.0f;
      input[index] = value;
      square_mean += value * value;
    }
    square_mean /= static_cast<float>(kHiddenSize);
    const float inverse_rms = 1.0f / std::sqrt(square_mean + kEpsilon);
    for (int32_t dimension = 0; dimension < kHiddenSize; ++dimension) {
      const int64_t index = static_cast<int64_t>(row) * kHiddenSize + dimension;
      scale[dimension] = 0.0001f * static_cast<float>(dimension);
      shift[dimension] = 0.001f * static_cast<float>(dimension);
      expected[index] = input[index] * inverse_rms * (1.0f + scale[dimension]) + shift[dimension];
    }
  }
  const auto input_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto scale_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), scale);
  const auto shift_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), shift);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("ada_rms_norm_float", test::MakeAdaRmsNormFloatSource()));
  ASSERT_TRUE(Launch(dim3(kRowCount, 1, 1), dim3(256, 1, 1), 0, input_device->device_address(),
                     scale_device->device_address(), shift_device->device_address(), kRowCount,
                     kHiddenSize, kEpsilon, output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-4f));
}
} // namespace
} // namespace omni_runtime::kernels
