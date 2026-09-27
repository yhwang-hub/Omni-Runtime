#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Omni-Runtime/inference/common/inference_define/inference_define.h"
#include "Omni-Runtime/memory/buffer.h"

namespace omni_runtime {
namespace inference {

class Tokenizer;

namespace qwen3_vl {

struct Qwen3VLSpecialTokens final {
  int32_t end_of_sequence_id = 151645;
  int32_t end_of_text_id = 151643;
  int32_t image_id = 151655;
  int32_t vision_start_id = 151652;
  int32_t vision_end_id = 151653;
};

struct Qwen3VLConfig final {
  int32_t layer_count = 0;
  int32_t hidden_size = 0;
  int32_t feed_forward_size = 0;
  int32_t query_head_count = 0;
  int32_t key_value_head_count = 0;
  int32_t head_size = 0;
  int32_t vocabulary_size = 0;
  int32_t deepstack_layer_count = 0;
  int32_t merge_size = 2;
  float rope_frequency_base = 0.0F;
  float rms_norm_epsilon = 0.0F;
  std::vector<int32_t> mrope_sections;
  Qwen3VLSpecialTokens special_tokens;
};

struct Qwen3VLVisionConfig final {
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

struct Qwen3VLVisionFeatures final {
  memory::BufferPtr embedding;
  std::vector<memory::BufferPtr> deepstack_embeddings;
  ImageSize input_image_size;
  ImageSize resized_image_size;
  int32_t token_count = 0;
  int32_t merged_grid_height = 0;
  int32_t merged_grid_width = 0;
};

struct Qwen3VLLanguageWorkspaces final {
  memory::BufferPtr tokens;
  memory::BufferPtr positions;
  memory::BufferPtr hidden;
  memory::BufferPtr next_hidden;
  memory::BufferPtr normalized;
  memory::BufferPtr query;
  memory::BufferPtr key;
  memory::BufferPtr value;
  memory::BufferPtr query_normed;
  memory::BufferPtr key_normed;
  memory::BufferPtr attention;
  memory::BufferPtr projected;
  memory::BufferPtr gate;
  memory::BufferPtr up;
  memory::BufferPtr logits;
  memory::BufferPtr sample;
  memory::BufferPtr decode_partials;
};

struct Qwen3VLVisionWorkspaces final {
  memory::BufferPtr source;
  memory::BufferPtr patches;
  memory::BufferPtr patch_embedding;
  memory::BufferPtr hidden;
  memory::BufferPtr residual;
  memory::BufferPtr normalized;
  memory::BufferPtr qkv;
  memory::BufferPtr query;
  memory::BufferPtr key;
  memory::BufferPtr value;
  memory::BufferPtr attention;
  memory::BufferPtr projected;
  memory::BufferPtr feed_forward;
  memory::BufferPtr embedding;
  std::vector<memory::BufferPtr> deepstack;
};

struct Qwen3VLGenerationContext final {
  const GenerationRequest *request = nullptr;
  GenerationResult *result = nullptr;
  const Tokenizer *tokenizer = nullptr;
  const Qwen3VLConfig *model_config = nullptr;
  std::vector<int32_t> prompt_token_ids;
  std::vector<int32_t> prefill_token_ids;
  std::vector<int32_t> prefill_position_ids;
  Qwen3VLVisionFeatures vision_features;
  int32_t vision_token_offset = -1;
  int32_t next_token_id = -1;
  std::string model_prompt;
  std::string error_message;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
