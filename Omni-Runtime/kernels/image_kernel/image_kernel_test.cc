#include <cstdint>
#include <cuda_fp16.h>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

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
    const __half half_value = __float2half_rn(value);
    return *reinterpret_cast<const uint16_t *>(&half_value);
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
      const uint8_t pixel = source[(pixel_index * 3U) + 2U - channel];
      const float normalized = static_cast<float>(pixel) / 127.5f - 1.0f;
      expected[destination_offset + channel * kPatchArea] = HalfBits(normalized);
    }
  }
  const auto source_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), source);
  const auto destination_device =
      memory::Buffer::CreateDeviceBuffer<uint16_t>(backend(), expected.size());
  ASSERT_TRUE(Compile("qwen3_vl_preprocess", test::MakeQwen3VLPreprocessSource()));
  ASSERT_TRUE(Launch(dim3(4, 1, 1), dim3(256, 1, 1), 0, kPixelCount,
                     source_device->device_address(), kSourceWidthPixels, kSourceHeightPixels,
                     kPatchSize, destination_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<uint16_t> actual(expected.size());
  ASSERT_TRUE(destination_device->CopyDataToVector(&actual));
  EXPECT_TRUE(actual == expected);
}
} // namespace
} // namespace omni_runtime::kernels
