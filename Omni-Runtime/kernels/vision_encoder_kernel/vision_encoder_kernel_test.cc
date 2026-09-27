#include <cstdint>
#include <cmath>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/vision_encoder_kernel/vision_encoder_kernel_runtime.h"

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

template <typename T> class VisionPatchLowPrecisionTest : public test::KernelTestFixture {};
using PatchTypes = ::testing::Types<Float16, BFloat16>;
TYPED_TEST_SUITE(VisionPatchLowPrecisionTest, PatchTypes);

TYPED_TEST(VisionPatchLowPrecisionTest, AccumulatesBothFramesBeforeRoundingWithTileTails) {
  constexpr int32_t kPatches = 65;
  constexpr int32_t kColumns = 67;
  for (const int32_t inner : {35, 768}) {
    SCOPED_TRACE(inner);
    std::vector<TypeParam> input(kPatches * inner);
    std::vector<TypeParam> first_weight(kColumns * inner);
    std::vector<TypeParam> second_weight(kColumns * inner);
    std::vector<TypeParam> bias(kColumns);
    for (size_t index = 0U; index < input.size(); ++index) {
      input[index] = test::LowPrecision<TypeParam>((static_cast<int32_t>(index % 7U) - 3) / 32.0F);
    }
    for (size_t index = 0U; index < first_weight.size(); ++index) {
      first_weight[index] =
          test::LowPrecision<TypeParam>((static_cast<int32_t>(index % 11U) - 5) / 16.0F);
      second_weight[index] =
          test::LowPrecision<TypeParam>((static_cast<int32_t>(index % 13U) - 6) / 8.0F);
    }
    for (int32_t index = 0; index < kColumns; ++index) {
      bias[index] = test::LowPrecision<TypeParam>((index % 5 - 2) / 16.0F);
    }
    const auto x = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), input);
    const auto w0 = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), first_weight);
    const auto w1 = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), second_weight);
    const auto b = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), bias);
    const auto y =
        memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), kPatches * kColumns);
    const int32_t patch = inner == 768 ? 16 : 1;
    ASSERT_TRUE((runtime::VisionPatchEmbedPairKernelRuntime<TypeParam>::CompileAndLaunch(
        "vision_patch_pair_low_precision", {.options = {},
                                            .input = x->device_address(),
                                            .first_weight = w0->device_address(),
                                            .second_weight = w1->device_address(),
                                            .bias = b->device_address(),
                                            .height = 5 * patch,
                                            .width = 13 * patch,
                                            .channels = inner / (patch * patch),
                                            .output_channels = kColumns,
                                            .spatial_patch = patch,
                                            .output = y->device_address()})));
    std::vector<TypeParam> actual(kPatches * kColumns);
    ASSERT_TRUE(y->CopyToHost(actual.size() * sizeof(TypeParam), actual.data()));
    for (int32_t row = 0; row < kPatches; ++row) {
      for (int32_t column = 0; column < kColumns; ++column) {
        float sum = test::HostFloat(bias[column]);
        for (int32_t d = 0; d < inner; ++d) {
          sum += test::HostFloat(input[row * inner + d]) *
                 (test::HostFloat(first_weight[column * inner + d]) +
                  test::HostFloat(second_weight[column * inner + d]));
        }
        EXPECT_FLOAT_EQ(test::HostFloat(actual[row * kColumns + column]),
                        test::HostFloat(test::LowPrecision<TypeParam>(sum)));
      }
    }
  }
}

TEST_F(VisionEncoderKernelTest, DistinguishesBlockAndMergerActivations) {
  const std::vector<float> input = {-5.0F, -3.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F, 5.0F};
  const std::vector<float> bias = {0.125F, -0.25F, 0.5F, 0.0F};
  const auto input_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto bias_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), bias);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), input.size());
  for (const bool is_approximate : {false, true}) {
    ASSERT_TRUE((runtime::VisionGeluBiasKernelRuntime<float>::CompileAndLaunch(
        "vision_gelu", {.options = {.grid_dim = {1U, 1U, 1U},
                                    .block_dim = {256U, 1U, 1U},
                                    .shared_memory_bytes = 0},
                        .input = input_device->device_address(),
                        .bias = bias_device->device_address(),
                        .element_count = input.size(),
                        .row_size = static_cast<int32_t>(bias.size()),
                        .is_approximate = is_approximate,
                        .output = output_device->device_address()})));
    std::vector<float> actual(input.size());
    ASSERT_TRUE(output_device->CopyToHost(actual.size() * sizeof(float), actual.data()));
    for (size_t index = 0U; index < input.size(); ++index) {
      const float value = input[index] + bias[index % bias.size()];
      const float activation =
          is_approximate
              ? std::tanh(0.7978845608028654F * (value + 0.044715F * value * value * value))
              : std::erf(value * 0.7071067811865475F);
      EXPECT_NEAR(actual[index], 0.5F * value * (1.0F + activation), 2.0e-6F);
    }
  }
}

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
  ASSERT_TRUE((runtime::VisionPatchEmbedKernelRuntime<float>::CompileAndLaunch(
      "vision_patch_embed_float",
      {.options = {.grid_dim = dim3(kPatchCount * kOutputChannels, 1, 1),
                   .block_dim = dim3(128, 1, 1),
                   .shared_memory_bytes = 0},
       .input = input_device->device_address(),
       .weight = weight_device->device_address(),
       .bias = bias_device->device_address(),
       .frames = kFrames,
       .height = kHeight,
       .width = kWidth,
       .channels = kChannels,
       .output_channels = kOutputChannels,
       .temporal_patch = kTemporalPatch,
       .spatial_patch = kSpatialPatch,
       .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels
