#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/image_kernel/image_kernel_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr uint64_t kPixelCount = 1024ULL;
constexpr int32_t kSourceWidthPixels = 32;
constexpr int32_t kSourceHeightPixels = 32;
constexpr int32_t kPatchSize = 16;
constexpr int32_t kPatchArea = kPatchSize * kPatchSize;
constexpr std::size_t kDestinationCount = 3U * kPatchArea * 4U;

class ImageKernelTest : public test::KernelTestFixture {
protected:
  static uint16_t HalfBits(const float value) {
    return test::LowPrecision<Float16>(value).bits;
  }
};

TEST_F(ImageKernelTest, PreprocessesQwen3VlImagePatch) {
  std::vector<uint8_t> source(3U * kPixelCount);
  std::vector<uint16_t> expected(kDestinationCount);
  for (std::size_t index = 0; index < source.size(); ++index) {
    source[index] = static_cast<uint8_t>((index * 31U) % 251U + 2U);
  }
  for (uint64_t pixel_index = 0; pixel_index < kPixelCount; ++pixel_index) {
    const int32_t row = static_cast<int32_t>(pixel_index / kSourceWidthPixels);
    const int32_t column = static_cast<int32_t>(pixel_index % kSourceWidthPixels);
    const int32_t patch_count_x = kSourceWidthPixels / kPatchSize;
    const int32_t patch_index = (row / kPatchSize) * patch_count_x + column / kPatchSize;
    const int32_t local_index = (row % kPatchSize) * kPatchSize + column % kPatchSize;
    const int64_t destination_offset =
        static_cast<int64_t>(patch_index) * 3 * kPatchArea + local_index;
    for (int32_t channel = 0; channel < 3; ++channel) {
      const uint8_t pixel = source[(pixel_index * 3U) + channel];
      const float normalized = static_cast<float>(pixel) / 127.5f - 1.0f;
      expected[destination_offset + channel * kPatchArea] = HalfBits(normalized);
    }
  }
  const auto source_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), source);
  const auto destination_device =
      memory::Buffer::CreateDeviceBuffer<uint16_t>(backend(), expected.size());
  ASSERT_TRUE((runtime::Qwen3VLPreprocessKernelRuntime<Float16>::CompileAndLaunch(
      "qwen3_vl_preprocess_half", {.options = {.grid_dim = dim3(4, 1, 1),
                                               .block_dim = dim3(256, 1, 1),
                                               .shared_memory_bytes = 0},
                                   .pixel_count = kPixelCount,
                                   .source = source_device->device_address(),
                                   .source_width_pixels = kSourceWidthPixels,
                                   .source_height_pixels = kSourceHeightPixels,
                                   .patch_size = kPatchSize,
                                   .destination = destination_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<uint16_t> actual(expected.size());
  ASSERT_TRUE(destination_device->CopyDataToVector(&actual));
  EXPECT_TRUE(actual == expected);
}

template <typename T> class ImageResizeLowPrecisionTest : public test::KernelTestFixture {};
using ImageTypes = ::testing::Types<Float16, BFloat16>;
TYPED_TEST_SUITE(ImageResizeLowPrecisionTest, ImageTypes);

TYPED_TEST(ImageResizeLowPrecisionTest, MatchesTorchvisionUint8BicubicAndPatchOrder) {
  struct ResizeCase final {
    int32_t input_size;
    std::vector<uint8_t> resized;
  };
  // Torchvision uint8 bicubic, antialias=True; input[i] = (37 * i + 11) % 256.
  const std::vector<ResizeCase> cases = {
      {3, {1,   38, 74,  68,  124, 170, 172, 109, 155, 249, 2,   38,  47,  84,  131, 140,
           183, 96, 167, 168, 81,  112, 48,  95,  121, 158, 205, 137, 172, 85,  114, 157,
           70,  66, 122, 169, 178, 215, 251, 61,  98,  144, 46,  83,  129, 142, 179, 215}},
      {5, {34,  82,  122, 161, 106, 151, 131, 77,  120, 190, 188, 56,  98,  138, 148, 118,
           168, 131, 91,  136, 189, 162, 65,  112, 154, 152, 169, 181, 112, 85,  160, 206,
           143, 79,  128, 126, 156, 178, 49,  107, 105, 160, 85,  126, 123, 93,  130, 152}}};
  for (const ResizeCase &resize_case : cases) {
    SCOPED_TRACE(resize_case.input_size);
    std::vector<uint8_t> source(resize_case.input_size * resize_case.input_size * 3);
    for (size_t index = 0U; index < source.size(); ++index) {
      source[index] = static_cast<uint8_t>((index * 37U + 11U) % 256U);
    }
    std::vector<TypeParam> expected(48U);
    for (int32_t row = 0; row < 4; ++row) {
      for (int32_t column = 0; column < 4; ++column) {
        const int32_t patch = (row / 2) * 2 + column / 2;
        for (int32_t channel = 0; channel < 3; ++channel) {
          const float pixel = resize_case.resized[(row * 4 + column) * 3 + channel];
          expected[patch * 12 + channel * 4 + (row % 2) * 2 + column % 2] =
              test::LowPrecision<TypeParam>((pixel - 127.5F) / 127.5F);
        }
      }
    }
    const auto input = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), source);
    const auto output =
        memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), expected.size());
    ASSERT_TRUE((runtime::Qwen3VLPreprocessKernelRuntime<TypeParam>::CompileAndLaunch(
        "image_resize_low_precision", {.options = {.grid_dim = {1U, 1U, 1U},
                                                   .block_dim = {256U, 1U, 1U},
                                                   .shared_memory_bytes = 0},
                                       .pixel_count = 16U,
                                       .source = input->device_address(),
                                       .source_width_pixels = 4,
                                       .source_height_pixels = 4,
                                       .patch_size = 2,
                                       .destination = output->device_address(),
                                       .input_width_pixels = resize_case.input_size,
                                       .input_height_pixels = resize_case.input_size})));
    std::vector<TypeParam> actual(expected.size());
    ASSERT_TRUE(output->CopyToHost(actual.size() * sizeof(TypeParam), actual.data()));
    for (size_t index = 0U; index < actual.size(); ++index) {
      EXPECT_EQ(actual[index].bits, expected[index].bits) << index;
    }
  }
}
} // namespace
} // namespace omni_runtime::kernels
