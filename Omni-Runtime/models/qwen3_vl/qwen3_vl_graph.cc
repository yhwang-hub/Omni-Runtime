#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_pipeline.h"

#include <algorithm>

#include "Omni-Runtime/inference/common/image_preprocessor/image_preprocessor.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

OperatorGraphBindings Qwen3VLPipeline::MakeBindings() const {
  OperatorGraphBindings bindings;
  bindings.resolve_view = [this](const std::string &name, backend::DeviceView *const view) {
    OMNI_RETURN_VAL_IF(view == nullptr || runtime_ == nullptr, false);
    *view = Weight(*runtime_, name);
    return view->IsValid();
  };
  return bindings;
}

bool Qwen3VLPipeline::BuildLanguageBindings(const int32_t token_count, const bool is_decode,
                                            const GenerationContext &context,
                                            OperatorGraphBindings *const bindings) const {
  OMNI_RETURN_VAL_IF_LOG(bindings == nullptr || runtime_ == nullptr || token_count <= 0, false,
                         ERROR, "Invalid language graph bindings");
  const Qwen3VLConfig &config = runtime_->model_config;
  OMNI_RETURN_VAL_IF_LOG(config.mrope_sections.size() < 3U, false, ERROR,
                         "Qwen3-VL requires three RoPE sections");
  const auto &workspace = runtime_->language_workspaces;
  *bindings = MakeBindings();
  bindings->values = {
      {"tokens", token_count},
      {"language_layers", config.layer_count},
      {"hidden_size", config.hidden_size},
      {"query_size", config.query_head_count * config.head_size},
      {"kv_size", config.key_value_head_count * config.head_size},
      {"ffn_size", config.feed_forward_size},
      {"query_heads", config.query_head_count},
      {"kv_heads", config.key_value_head_count},
      {"head_size", config.head_size},
      {"query_rows", token_count * config.query_head_count},
      {"kv_rows", token_count * config.key_value_head_count},
      {"capacity", runtime_->cache_capacity},
      {"start_position", runtime_->used_token_count},
      {"cached_tokens",
       is_decode ? runtime_->cache_capacity : token_count + runtime_->used_token_count},
      {"epsilon", config.rms_norm_epsilon},
      {"rope_base", config.rope_frequency_base},
      {"rope_t", config.mrope_sections[0]},
      {"rope_h", config.mrope_sections[1]},
      {"rope_w", config.mrope_sections[2]},
      {"vocabulary", config.vocabulary_size},
      {"hidden_values", static_cast<uint64_t>(token_count) * config.hidden_size},
      {"ffn_values", static_cast<uint64_t>(token_count) * config.feed_forward_size},
      {"vision_values",
       static_cast<uint64_t>(context.vision_features.token_count) * config.hidden_size},
      {"is_prefill", !is_decode},
      {"is_decode", is_decode},
      {"has_deepstack", false},
      {"token_input", is_decode ? workspace.sample->view() : workspace.tokens->view()},
      {"hidden", workspace.hidden->view()},
      {"next_hidden", workspace.next_hidden->view()},
      {"normalized", workspace.normalized->view()},
      {"query", workspace.query->view()},
      {"key", workspace.key->view()},
      {"value", workspace.value->view()},
      {"query_normed", workspace.query_normed->view()},
      {"key_normed", workspace.key_normed->view()},
      {"positions", workspace.positions->view()},
      {"attention", workspace.attention->view()},
      {"projected", workspace.projected->view()},
      {"gate", workspace.gate->view()},
      {"up", workspace.up->view()},
      {"logits", workspace.logits->view()},
      {"sample", workspace.sample->view()},
      {"decode_partials", workspace.decode_partials->view()},
      {"cache_position", runtime_->decode_cache_position->view()},
      {"attention_position",
       is_decode ? runtime_->decode_cache_position->view() : backend::DeviceView{}},
      {"output_weight", Weight(*runtime_, weights_.output_weight().name)}};
  const size_t row_bytes = static_cast<size_t>(config.hidden_size) * RuntimeElementSize(*runtime_);
  bindings->values["last_hidden"] =
      workspace.hidden->view().Slice(static_cast<size_t>(token_count - 1) * row_bytes, row_bytes);
  if (context.vision_features.embedding != nullptr) {
    bindings->values["vision_hidden"] = workspace.hidden->view().Slice(
        static_cast<size_t>(context.vision_token_offset) * row_bytes,
        static_cast<size_t>(context.vision_features.token_count) * row_bytes);
  }
  return true;
}

bool Qwen3VLPipeline::BindLanguageLayer(const std::string &variable, const int32_t layer,
                                        const GenerationContext &context, const bool is_decode,
                                        OperatorGraphBindings *const bindings) const {
  OMNI_RETURN_VAL_IF(bindings == nullptr || runtime_ == nullptr, false);
  if (variable != "layer") {
    return true;
  }
  OMNI_RETURN_VAL_IF(layer < 0 || layer >= runtime_->model_config.layer_count, false);
  const size_t cache_bytes = static_cast<size_t>(runtime_->cache_capacity) *
                             runtime_->model_config.key_value_head_count *
                             runtime_->model_config.head_size * RuntimeElementSize(*runtime_);
  bindings->values["key_cache"] =
      runtime_->key_cache->view().Slice(static_cast<size_t>(layer) * cache_bytes, cache_bytes);
  bindings->values["value_cache"] =
      runtime_->value_cache->view().Slice(static_cast<size_t>(layer) * cache_bytes, cache_bytes);
  const bool has_deepstack = !is_decode && static_cast<size_t>(layer) <
                                               context.vision_features.deepstack_embeddings.size();
  bindings->values["has_deepstack"] = has_deepstack;
  if (has_deepstack) {
    bindings->values["deepstack_embedding"] =
        context.vision_features.deepstack_embeddings[layer]->view();
  }
  return true;
}

bool Qwen3VLPipeline::BuildGraph() {
  const GenerationContext context;
  OperatorGraphBindings bindings;
  OMNI_RETURN_VAL_IF(!BuildLanguageBindings(1, true, context, &bindings),
                     SetError("failed to bind Qwen3-VL decode graph"));
  const OperatorIterationBinder bind_layer =
      [this, &context](const std::string &variable, const int32_t layer,
                       OperatorGraphBindings *const iteration) {
        return BindLanguageLayer(variable, layer, context, true, iteration);
      };
  std::vector<backend::ComputeNode> nodes;
  for (const std::string section : {"embedding", "language", "decode"}) {
    OMNI_RETURN_VAL_IF(!graph_config_.Append(section, bindings, bind_layer, &nodes),
                       SetError(graph_config_.error_message()));
  }
  OMNI_RETURN_VAL_IF(
      !runtime_->backend->CreateExecutionPlan(RuntimeScalarType(*runtime_), nodes,
                                              runtime_->language_workspaces.sample->allocation(),
                                              &runtime_->decode_plan),
      SetError("failed to prepare Qwen3-VL decode operator graph"));
  return true;
}

bool Qwen3VLPipeline::RunVision(const GenerationRequest &request, GenerationContext *const context,
                                GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(context == nullptr || result == nullptr, SetError("vision outputs are null"));
  if (request.image_path.empty()) {
    return true;
  }
  std::vector<uint8_t> pixels;
  int32_t width_pixels = 0;
  int32_t height_pixels = 0;
  ImagePatchGrid grid;
  const auto &config = runtime_->vision_config;
  OMNI_RETURN_VAL_IF(
      !ImagePreprocessor::LoadImage(request.image_path, &pixels, &width_pixels, &height_pixels) ||
          !ImagePreprocessor::MakeGrid(width_pixels, height_pixels, config.patch_size,
                                       config.merge_size, &grid),
      SetError("failed to prepare Qwen3-VL image"));
  OMNI_RETURN_VAL_IF(grid.merged_token_count > runtime_->cache_capacity,
                     SetError("Qwen3-VL image exceeds workspace capacity"));
  const auto &workspace = runtime_->vision_workspaces;
  OperatorGraphBindings bindings = MakeBindings();
  bindings.values = {
      {"vision_layers", config.layer_count},
      {"v_tokens", grid.token_count},
      {"v_hidden", config.hidden_size},
      {"v_qkv", config.hidden_size * 3},
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
      {"resized_width", grid.width_patches * config.patch_size},
      {"resized_height", grid.height_patches * config.patch_size},
      {"input_width", width_pixels},
      {"input_height", height_pixels},
      {"pixels", static_cast<uint64_t>(grid.width_patches) * grid.height_patches *
                     config.patch_size * config.patch_size},
      {"source", workspace.source->view()},
      {"patches", workspace.patches->view()},
      {"patch_embedding", workspace.patch_embedding->view()},
      {"hidden", workspace.hidden->view()},
      {"residual", workspace.residual->view()},
      {"normalized", workspace.normalized->view()},
      {"qkv", workspace.qkv->view()},
      {"query", workspace.query->view()},
      {"key", workspace.key->view()},
      {"value", workspace.value->view()},
      {"attention", workspace.attention->view()},
      {"projected", workspace.projected->view()},
      {"feed_forward", workspace.feed_forward->view()},
      {"embedding", workspace.embedding->view()},
      {"projector_first_weight", Weight(*runtime_, "mm.0.weight")},
      {"projector_first_bias", Weight(*runtime_, "mm.0.bias")},
      {"projector_second_weight", Weight(*runtime_, "mm.2.weight")},
      {"projector_second_bias", Weight(*runtime_, "mm.2.bias")}};
  const OperatorIterationBinder bind_layer =
      [this, &config, &workspace](const std::string &variable, const int32_t layer,
                                  OperatorGraphBindings *const iteration) {
        OMNI_RETURN_VAL_IF(iteration == nullptr, false);
        if (variable != "vision_layer") {
          return true;
        }
        const auto found =
            std::find(config.deepstack_layers.begin(), config.deepstack_layers.end(), layer);
        const bool has_deepstack = found != config.deepstack_layers.end();
        iteration->values["has_deepstack"] = has_deepstack;
        if (has_deepstack) {
          const size_t index =
              static_cast<size_t>(std::distance(config.deepstack_layers.begin(), found));
          iteration->values["embedding"] = workspace.deepstack[index]->view();
          const std::string prefix = "v.deepstack." + std::to_string(layer) + ".";
          iteration->values["projector_first_weight"] = Weight(*runtime_, prefix + "fc1.weight");
          iteration->values["projector_first_bias"] = Weight(*runtime_, prefix + "fc1.bias");
          iteration->values["projector_second_weight"] = Weight(*runtime_, prefix + "fc2.weight");
          iteration->values["projector_second_bias"] = Weight(*runtime_, prefix + "fc2.bias");
        }
        return true;
      };
  std::vector<backend::ComputeNode> nodes;
  OMNI_RETURN_VAL_IF(!graph_config_.Append("vision", bindings, bind_layer, &nodes),
                     SetError(graph_config_.error_message()));
  OMNI_RETURN_VAL_IF(!vision_graph_.Prepare("qwen3_vl/vision", runtime_->backend,
                                            RuntimeScalarType(*runtime_), nodes),
                     SetError(vision_graph_.error_message()));
  OMNI_RETURN_VAL_IF(
      !runtime_->compute_timer.Start() ||
          !workspace.source->CopyFromHost(pixels.data(), pixels.size()) || !vision_graph_.Run() ||
          !runtime_->compute_timer.Stop(&runtime_->step_time_ms),
      SetError("Qwen3-VL vision operator graph failed: " + vision_graph_.error_message()));
  auto &features = context->vision_features;
  features.embedding = workspace.embedding;
  features.deepstack_embeddings = workspace.deepstack;
  features.token_count = grid.merged_token_count;
  features.merged_grid_height = grid.height_patches / config.merge_size;
  features.merged_grid_width = grid.width_patches / config.merge_size;
  features.input_image_size = {.width_pixels = width_pixels, .height_pixels = height_pixels};
  features.resized_image_size = {.width_pixels = grid.width_patches * config.patch_size,
                                 .height_pixels = grid.height_patches * config.patch_size};
  result->vision_time_ms = runtime_->step_time_ms;
  result->input_image_size = features.input_image_size;
  result->resized_image_size = features.resized_image_size;
  return true;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
