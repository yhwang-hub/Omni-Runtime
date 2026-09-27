#pragma once

#include <cstdint>
#include <vector>

namespace omni_runtime::inference::qwen3 {

struct Qwen3VisionConfig final {
  int32_t image_size = 0;
  int32_t patch_size = 0;
  int32_t hidden_size = 0;
  int32_t feed_forward_size = 0;
  int32_t layer_count = 0;
  int32_t head_count = 0;
  int32_t projection_size = 0;
  int32_t merge_size = 0;
  float layer_norm_epsilon = 0.0F;
  std::vector<int32_t> deepstack_layers;

  int32_t head_size() const {
    return head_count == 0 ? 0 : hidden_size / head_count;
  }
  int32_t patch_alignment() const {
    return patch_size * merge_size;
  }
  int32_t merged_hidden_size() const {
    return hidden_size * merge_size * merge_size;
  }
  int32_t patch_value_count() const {
    return 3 * patch_size * patch_size;
  }
};

} // namespace omni_runtime::inference::qwen3
