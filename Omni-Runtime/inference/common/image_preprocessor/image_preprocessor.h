#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Omni-Runtime/inference/common/image_preprocessor/image_preprocessor_define.h"

namespace omni_runtime {
namespace inference {

class ImagePreprocessor final {
public:
  static bool LoadImage(const std::string &image_path, std::vector<uint8_t> *const pixels,
                        int32_t *const width_pixels, int32_t *const height_pixels);
  static bool MakeGrid(const int32_t source_width_pixels, const int32_t source_height_pixels,
                       const int32_t patch_size, const int32_t merge_size,
                       ImagePatchGrid *const grid);
};

} // namespace inference
} // namespace omni_runtime
