#pragma once

#include <cstdint>
#include <vector>

namespace omni_runtime {
namespace inference {
namespace qwen3 {

struct LanguageConfig {
  int32_t layer_count = 0;
  int32_t hidden_size = 0;
  int32_t feed_forward_size = 0;
  int32_t query_head_count = 0;
  int32_t key_value_head_count = 0;
  int32_t head_size = 0;
  int32_t vocabulary_size = 0;
  float rope_frequency_base = 0.0F;
  float rms_norm_epsilon = 0.0F;
  std::vector<int32_t> mrope_sections;
};

struct Prompt {
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  int32_t image_offset = -1;
  int32_t image_tokens = 0;
  int32_t image_grid_height = 0;
  int32_t image_grid_width = 0;
  int32_t next_position = 0;
};

} // namespace qwen3
} // namespace inference
} // namespace omni_runtime
