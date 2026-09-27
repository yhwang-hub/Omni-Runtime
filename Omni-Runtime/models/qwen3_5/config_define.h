#pragma once

#include <cstdint>
#include <vector>

#include "Omni-Runtime/models/components/qwen_vision/config_define.h"
#include "Omni-Runtime/models/components/qwen_multimodal/processor_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_5 {

struct Qwen35Config final : public qwen3::LanguageConfig {
  int32_t rotary_size = 0;
  int32_t convolution_size = 0;
  int32_t recurrent_head_size = 0;
  int32_t recurrent_key_heads = 0;
  int32_t recurrent_value_heads = 0;
  int32_t full_attention_interval = 0;
  int32_t end_of_sequence_id = -1;
  int32_t end_of_text_id = -1;
  int32_t image_id = -1;
  bool has_fp8_weights = false;
  std::vector<bool> recurrent_layers;
  qwen3::Qwen3VisionConfig vision;

  bool IsRecurrent(const int32_t layer) const {
    return layer >= 0 && static_cast<size_t>(layer) < recurrent_layers.size() &&
           recurrent_layers[layer];
  }
};

} // namespace qwen3_5
} // namespace inference
} // namespace omni_runtime
