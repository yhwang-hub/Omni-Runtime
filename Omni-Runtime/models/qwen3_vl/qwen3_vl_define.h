#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/inference/common/inference_define/inference_define.h"
#include "Omni-Runtime/memory/buffer.h"
#include "Omni-Runtime/memory/tensor.h"

namespace omni_runtime {
namespace util {

class GGUFSerializer;

} // namespace util

namespace inference {

class InferenceConfig;
namespace qwen3_vl {

class Qwen3VLWeightRegistry;

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

struct Qwen3VLWeightName final {
  std::string name;
  std::vector<uint64_t> dimensions;
};

struct Qwen3VLLayerWeightNames final {
  Qwen3VLWeightName attention_norm;
  Qwen3VLWeightName query_norm;
  Qwen3VLWeightName key_norm;
  Qwen3VLWeightName query;
  Qwen3VLWeightName key;
  Qwen3VLWeightName value;
  Qwen3VLWeightName attention_output;
  Qwen3VLWeightName feed_forward_norm;
  Qwen3VLWeightName feed_forward_gate;
  Qwen3VLWeightName feed_forward_up;
  Qwen3VLWeightName feed_forward_down;
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

struct Qwen3VLRuntime final {
  std::shared_ptr<backend::Backend> backend;
  backend::EventTimer compute_timer;
  double step_time_ms = 0.0;
  const InferenceConfig *config = nullptr;
  const util::GGUFSerializer *model_serializer = nullptr;
  const util::GGUFSerializer *vision_serializer = nullptr;
  const Qwen3VLWeightRegistry *registry = nullptr;
  Qwen3VLConfig model_config;
  Qwen3VLVisionConfig vision_config;
  std::unordered_map<std::string, memory::TensorPtr> weights;
  Qwen3VLLanguageWorkspaces language_workspaces;
  Qwen3VLVisionWorkspaces vision_workspaces;
  memory::BufferPtr key_cache;
  memory::BufferPtr value_cache;
  memory::BufferPtr decode_cache_position;
  std::unique_ptr<backend::ExecutionPlan> decode_plan;
  int32_t cache_capacity = 0;
  int32_t used_token_count = 0;
  int32_t next_rope_position = 0;
  size_t gpu_memory_bytes = 0U;
  bool has_weights_loaded = false;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
