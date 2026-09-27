#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Omni-Runtime/models/components/qwen_multimodal/processor_define.h"
#include "Omni-Runtime/models/components/qwen_vision/config_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

struct Qwen3VLConfig final : public qwen3::LanguageConfig {
  struct SpecialTokens final {
    int32_t end_of_sequence_id = 151645;
    int32_t end_of_text_id = 151643;
    int32_t image_id = 151655;
    int32_t vision_start_id = 151652;
    int32_t vision_end_id = 151653;
  };

  int32_t deepstack_layer_count = 0;
  int32_t merge_size = 2;
  SpecialTokens special_tokens;
};

struct Qwen3VLWeightName final {
  std::string name;
  std::vector<uint64_t> dimensions;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
