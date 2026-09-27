#pragma once

#include <cstdint>

namespace omni_runtime {
namespace inference {

struct ImagePatchGrid final {
  int32_t height_patches = 0;
  int32_t width_patches = 0;
  int32_t token_count = 0;
  int32_t merged_token_count = 0;
};

} // namespace inference
} // namespace omni_runtime
