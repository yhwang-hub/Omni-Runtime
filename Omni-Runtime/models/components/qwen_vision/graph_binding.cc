#include "Omni-Runtime/models/components/qwen_vision/graph_binding.h"

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3 {

bool BindVisionDimensions(const Qwen3VisionConfig &config, const ImagePatchGrid &grid,
                          const int32_t width_pixels, const int32_t height_pixels,
                          OperatorGraphBindings *const bindings) {
  OMNI_RETURN_VAL_IF_LOG(bindings == nullptr, false, ERROR, "Vision bindings are null");
  const std::pair<std::string, OperatorGraphBindings::Value> values[] = {
      {"vision_layers", config.layer_count},
      {"v_tokens", grid.token_count},
      {"v_hidden", config.hidden_size},
      {"v_qkv", 3 * config.hidden_size},
      {"v_ffn", config.feed_forward_size},
      {"v_heads", config.head_count},
      {"v_head_size", config.head_size()},
      {"v_epsilon", config.layer_norm_epsilon},
      {"v_hidden_values", static_cast<uint64_t>(grid.token_count) * config.hidden_size},
      {"v_qkv_values", static_cast<uint64_t>(grid.token_count) * config.hidden_size * 3U},
      {"v_ffn_values", static_cast<uint64_t>(grid.token_count) * config.feed_forward_size},
      {"merged_tokens", grid.merged_token_count},
      {"merged_hidden", config.merged_hidden_size()},
      {"merged_values",
       static_cast<uint64_t>(grid.merged_token_count) * config.merged_hidden_size()},
      {"embedding_values", static_cast<uint64_t>(grid.merged_token_count) * config.projection_size},
      {"projection_size", config.projection_size},
      {"grid_height", grid.height_patches},
      {"grid_width", grid.width_patches},
      {"patch_size", config.patch_size},
      {"merge_size", config.merge_size},
      {"position_grid_size", config.image_size / config.patch_size},
      {"resized_width", grid.width_patches * config.patch_size},
      {"resized_height", grid.height_patches * config.patch_size},
      {"input_width", width_pixels},
      {"input_height", height_pixels},
      {"pixels", static_cast<uint64_t>(grid.width_patches) * grid.height_patches *
                     config.patch_size * config.patch_size},
  };
  for (const auto &[name, value] : values) {
    bindings->values[name] = value;
  }
  return true;
}

} // namespace qwen3
} // namespace inference
} // namespace omni_runtime
