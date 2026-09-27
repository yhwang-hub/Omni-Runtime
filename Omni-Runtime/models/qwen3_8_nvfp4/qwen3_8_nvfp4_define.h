#pragma once

#include <cstdint>
#include <vector>

#include "Omni-Runtime/models/common/qwen3_vision_define.h"

namespace omni_runtime::inference::qwen3_8_nvfp4 {

struct Qwen38Config final {
  int32_t layer_count = 0;
  int32_t hidden_size = 0;
  int32_t feed_forward_size = 0;
  int32_t query_head_count = 0;
  int32_t key_value_head_count = 0;
  int32_t head_size = 0;
  int32_t rotary_size = 0;
  int32_t convolution_size = 0;
  int32_t recurrent_head_size = 0;
  int32_t recurrent_key_heads = 0;
  int32_t recurrent_value_heads = 0;
  int32_t full_attention_interval = 0;
  int32_t vocabulary_size = 0;
  int32_t end_of_sequence_id = -1;
  int32_t end_of_text_id = -1;
  int32_t image_id = -1;
  float rope_frequency_base = 0.0F;
  float rms_norm_epsilon = 0.0F;
  std::vector<int32_t> mrope_sections;
  std::vector<bool> recurrent_layers;
  qwen3::Qwen3VisionConfig vision;

  bool IsRecurrent(const int32_t layer) const {
    return layer >= 0 && static_cast<size_t>(layer) < recurrent_layers.size() &&
           recurrent_layers[layer];
  }
};

} // namespace omni_runtime::inference::qwen3_8_nvfp4
