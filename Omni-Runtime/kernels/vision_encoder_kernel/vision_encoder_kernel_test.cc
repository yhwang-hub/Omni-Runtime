#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kFrames = 2;
constexpr int32_t kHeight = 16;
constexpr int32_t kWidth = 16;
constexpr int32_t kChannels = 3;
constexpr int32_t kOutputChannels = 1152;
constexpr int32_t kTemporalPatch = 2;
constexpr int32_t kSpatialPatch = 16;
constexpr int32_t kPatchVolume = kChannels * kTemporalPatch * kSpatialPatch * kSpatialPatch;
constexpr int32_t kPatchCount = 1;

class VisionEncoderKernelTest : public test::KernelTestFixture {};

TEST_F(VisionEncoderKernelTest, EmbedsQwen3VlVisionPatch) {
  std::vector<float> input(static_cast<std::size_t>(kFrames) * kHeight * kWidth * kChannels);
  std::vector<float> weight(static_cast<std::size_t>(kOutputChannels) * kPatchVolume);
  std::vector<float> bias(kOutputChannels);
  std::vector<float> expected(kOutputChannels);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = 0.001f * static_cast<float>(index) - 0.768f;
  }
  for (int32_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
    bias[output_channel] = 0.0001f * static_cast<float>(output_channel);
    float value = bias[output_channel];
    for (int32_t patch_index = 0; patch_index < kPatchVolume; ++patch_index) {
      const float weight_value = 0.000001f * static_cast<float>(patch_index) +
                                 0.0000001f * static_cast<float>(output_channel);
      weight[static_cast<int64_t>(output_channel) * kPatchVolume + patch_index] = weight_value;
      value += input[patch_index] * weight_value;
    }
    expected[output_channel] = value;
  }
  const auto input_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto weight_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), weight);
  const auto bias_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), bias);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("vision_patch_embed_float", test::MakeVisionPatchEmbedFloatSource()));
  ASSERT_TRUE(Launch(
      dim3(kPatchCount * kOutputChannels, 1, 1), dim3(128, 1, 1), 0, input_device->device_address(),
      weight_device->device_address(), bias_device->device_address(), kFrames, kHeight, kWidth,
      kChannels, kOutputChannels, kTemporalPatch, kSpatialPatch, output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels
