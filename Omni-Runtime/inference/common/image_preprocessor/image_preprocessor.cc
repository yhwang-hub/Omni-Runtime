#include "Omni-Runtime/inference/common/image_preprocessor/image_preprocessor.h"

#include <cstdio>
#include <limits>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Omni-Runtime/inference/common/image_preprocessor/jpeg_decoder.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {

bool ImagePreprocessor::LoadImage(const std::string &image_path, std::vector<uint8_t> *const pixels,
                                  int32_t *const width_pixels, int32_t *const height_pixels) {
  return LoadImage(image_path, false, pixels, width_pixels, height_pixels);
}

bool ImagePreprocessor::LoadImage(const std::string &image_path, const bool use_stb_decoder,
                                  std::vector<uint8_t> *const pixels, int32_t *const width_pixels,
                                  int32_t *const height_pixels) {
  OMNI_RETURN_VAL_IF_LOG(pixels == nullptr || width_pixels == nullptr || height_pixels == nullptr,
                         false, ERROR, "Image output is null");
  const std::unique_ptr<FILE, int (*)(FILE *)> file(std::fopen(image_path.c_str(), "rb"),
                                                    &std::fclose);
  OMNI_RETURN_VAL_IF_LOG(file == nullptr, false, ERROR, "Cannot open image: " << image_path);
  const int first = std::fgetc(file.get());
  const int second = std::fgetc(file.get());
  OMNI_RETURN_VAL_IF_LOG(std::fseek(file.get(), 0L, SEEK_SET) != 0, false, ERROR,
                         "Cannot rewind image: " << image_path);
  if (!use_stb_decoder && first == 0xFF && second == 0xD8) {
    OMNI_RETURN_VAL_IF_LOG(!JpegDecoder::Load(file.get(), pixels, width_pixels, height_pixels),
                           false, ERROR, "JPEG image decode failed: " << image_path);
    return true;
  }
  int source_width = 0;
  int source_height = 0;
  int channel_count = 0;
  const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> source(
      stbi_load_from_file(file.get(), &source_width, &source_height, &channel_count, 3),
      &stbi_image_free);
  OMNI_RETURN_VAL_IF_LOG(source == nullptr || source_width <= 0 || source_height <= 0, false, ERROR,
                         "Image decode failed: " << image_path);
  const uint64_t size_bytes = static_cast<uint64_t>(source_width) * source_height * 3U;
  OMNI_RETURN_VAL_IF_LOG(size_bytes > pixels->max_size(), false, ERROR,
                         "Image dimensions exceed RGB storage capacity: " << image_path);
  pixels->assign(source.get(), source.get() + static_cast<size_t>(size_bytes));
  *width_pixels = source_width;
  *height_pixels = source_height;
  return true;
}

bool ImagePreprocessor::MakeGrid(const int32_t source_width_pixels,
                                 const int32_t source_height_pixels, const int32_t patch_size,
                                 const int32_t merge_size, ImagePatchGrid *const grid) {
  OMNI_RETURN_VAL_IF_LOG(grid == nullptr || source_width_pixels <= 0 || source_height_pixels <= 0 ||
                             patch_size <= 0 || merge_size <= 0,
                         false, ERROR, "Invalid image patch grid inputs");
  const int64_t alignment = static_cast<int64_t>(patch_size) * merge_size;
  const auto aligned_size = [alignment](const int32_t size_pixels) {
    return (size_pixels + alignment - 1) / alignment * alignment;
  };
  const int64_t width_pixels = aligned_size(source_width_pixels);
  const int64_t height_pixels = aligned_size(source_height_pixels);
  const int64_t max_dimension = std::numeric_limits<int32_t>::max();
  OMNI_RETURN_VAL_IF_LOG(width_pixels > max_dimension || height_pixels > max_dimension, false,
                         ERROR, "Aligned image dimensions exceed the supported range");
  const int64_t width_patches = width_pixels / patch_size;
  const int64_t height_patches = height_pixels / patch_size;
  const int64_t token_count = width_patches * height_patches;
  OMNI_RETURN_VAL_IF_LOG(token_count > max_dimension, false, ERROR,
                         "Image patch token count exceeds the supported range");
  *grid = {.height_patches = static_cast<int32_t>(height_patches),
           .width_patches = static_cast<int32_t>(width_patches),
           .token_count = static_cast<int32_t>(token_count),
           .merged_token_count =
               static_cast<int32_t>(token_count / (static_cast<int64_t>(merge_size) * merge_size))};
  return true;
}

} // namespace inference
} // namespace omni_runtime
