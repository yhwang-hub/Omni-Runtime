#include "Omni-Runtime/models/qwen3_vl/executor.h"
#include "Omni-Runtime/models/qwen3_vl/model.h"

#include <algorithm>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

backend::DeviceView Qwen3VLExecutor::View(const std::string &name) const {
  const auto iterator = model_.data().weights.find(name);
  return iterator == model_.data().weights.end() ? backend::DeviceView{}
                                                 : iterator->second->buffer()->view();
}

bool Qwen3VLExecutor::BuildLanguageBindings(const int32_t token_count, const bool is_decode,
                                            const GenerationContext &context,
                                            OperatorGraphBindings *const bindings) const {
  OMNI_RETURN_VAL_IF_LOG(bindings == nullptr || context_ == nullptr || token_count <= 0, false,
                         ERROR, "Invalid language graph bindings");
  const Qwen3VLConfig &config = model_.data().model_config;
  OMNI_RETURN_VAL_IF_LOG(config.mrope_sections.size() < 3U, false, ERROR,
                         "Qwen3-VL requires three RoPE sections");
  const auto &workspace = context_->language_workspaces;
  *bindings = MakeOperatorGraphBindings([this](const std::string &name) { return View(name); });
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
      {"capacity", state_.cache_capacity},
      {"start_position", state_.used_token_count},
      {"cached_tokens", is_decode ? state_.cache_capacity : token_count + state_.used_token_count},
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
      {"cache_position", state_.decode_cache_position->view()},
      {"attention_position",
       is_decode ? state_.decode_cache_position->view() : backend::DeviceView{}},
      {"output_weight", View(model_.output_weight_name())}};
  const size_t row_bytes = static_cast<size_t>(config.hidden_size) * RuntimeElementSize();
  bindings->values["last_hidden"] =
      workspace.hidden->view().Slice(static_cast<size_t>(token_count - 1) * row_bytes, row_bytes);
  if (context.vision_features.embedding != nullptr) {
    bindings->values["vision_hidden"] = workspace.hidden->view().Slice(
        static_cast<size_t>(context.image_offset) * row_bytes,
        static_cast<size_t>(context.vision_features.token_count) * row_bytes);
  }
  return true;
}

bool Qwen3VLExecutor::BindLanguageLayer(const std::string &variable, const int32_t layer,
                                        const GenerationContext &context, const bool is_decode,
                                        OperatorGraphBindings *const bindings) const {
  OMNI_RETURN_VAL_IF(bindings == nullptr || context_ == nullptr, false);
  if (variable != "layer") {
    return true;
  }
  OMNI_RETURN_VAL_IF(layer < 0 || layer >= model_.data().model_config.layer_count, false);
  const size_t cache_bytes = static_cast<size_t>(state_.cache_capacity) *
                             model_.data().model_config.key_value_head_count *
                             model_.data().model_config.head_size * RuntimeElementSize();
  bindings->values["key_cache"] =
      state_.key_cache->view().Slice(static_cast<size_t>(layer) * cache_bytes, cache_bytes);
  bindings->values["value_cache"] =
      state_.value_cache->view().Slice(static_cast<size_t>(layer) * cache_bytes, cache_bytes);
  const bool has_deepstack = !is_decode && static_cast<size_t>(layer) <
                                               context.vision_features.deepstack_embeddings.size();
  bindings->values["has_deepstack"] = has_deepstack;
  if (has_deepstack) {
    bindings->values["deepstack_embedding"] =
        context.vision_features.deepstack_embeddings[layer]->view();
  }
  return true;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
