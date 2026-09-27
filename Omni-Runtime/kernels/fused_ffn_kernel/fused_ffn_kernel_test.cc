#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kRows = 1;
constexpr int32_t kInputSize = 4096;
constexpr int32_t kIntermediateSize = 12288;
constexpr int32_t kOutputSize = 4096;

class FusedFfnKernelTest : public test::KernelTestFixture {
protected:
  static float GeluTanh(const float value) {
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    return 0.5f * value * (1.0f + std::tanh(kAlpha * (value + kBeta * value * value * value)));
  }
};

TEST_F(FusedFfnKernelTest, TransformsQwen3VlFeedForward) {
  std::vector<float> input(kInputSize);
  std::vector<float> fc1_weight(static_cast<std::size_t>(kIntermediateSize) * kInputSize);
  std::vector<float> fc1_bias(kIntermediateSize);
  std::vector<float> fc2_weight(static_cast<std::size_t>(kOutputSize) * kIntermediateSize);
  std::vector<float> fc2_bias(kOutputSize);
  std::vector<float> expected(kOutputSize);
  for (int32_t index = 0; index < kInputSize; ++index) {
    input[index] = 0.0001f * static_cast<float>(index) - 0.2f;
  }
  for (int32_t neuron = 0; neuron < kIntermediateSize; ++neuron) {
    fc1_bias[neuron] = 0.00001f * static_cast<float>(neuron);
    for (int32_t index = 0; index < kInputSize; ++index) {
      fc1_weight[static_cast<int64_t>(neuron) * kInputSize + index] =
          0.000001f * static_cast<float>(index) + 0.0000001f * static_cast<float>(neuron);
    }
  }
  for (int32_t neuron = 0; neuron < kOutputSize; ++neuron) {
    fc2_bias[neuron] = 0.00001f * static_cast<float>(neuron);
    for (int32_t index = 0; index < kIntermediateSize; ++index) {
      fc2_weight[static_cast<int64_t>(neuron) * kIntermediateSize + index] =
          0.0000001f * static_cast<float>(index) + 0.00000001f * static_cast<float>(neuron);
    }
  }
  std::vector<float> intermediate(kIntermediateSize);
  for (int32_t neuron = 0; neuron < kIntermediateSize; ++neuron) {
    float value = fc1_bias[neuron];
    for (int32_t index = 0; index < kInputSize; ++index) {
      value += input[index] * fc1_weight[static_cast<int64_t>(neuron) * kInputSize + index];
    }
    intermediate[neuron] = GeluTanh(value);
  }
  for (int32_t neuron = 0; neuron < kOutputSize; ++neuron) {
    float value = fc2_bias[neuron];
    for (int32_t index = 0; index < kIntermediateSize; ++index) {
      value += intermediate[index] *
               fc2_weight[static_cast<int64_t>(neuron) * kIntermediateSize + index];
    }
    expected[neuron] = value;
  }
  const auto input_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto fc1_weight_device =
      memory::Buffer::CreateDeviceBufferFromVector(backend(), fc1_weight);
  const auto fc1_bias_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), fc1_bias);
  const auto fc2_weight_device =
      memory::Buffer::CreateDeviceBufferFromVector(backend(), fc2_weight);
  const auto fc2_bias_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), fc2_bias);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("fused_ffn_float", test::MakeFusedFfnFloatSource()));
  const int shared_memory_bytes = (kInputSize + kIntermediateSize) * sizeof(float);
  ASSERT_TRUE(Launch(dim3(kRows, 1, 1), dim3(256, 1, 1), shared_memory_bytes,
                     input_device->device_address(), fc1_weight_device->device_address(),
                     fc1_bias_device->device_address(), fc2_weight_device->device_address(),
                     fc2_bias_device->device_address(), kRows, kInputSize, kIntermediateSize,
                     kOutputSize, output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-4f));
}
} // namespace
} // namespace omni_runtime::kernels
