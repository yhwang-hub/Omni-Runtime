#include "Omni-Runtime/models/qwen3_5/executor.h"
#include "Omni-Runtime/models/qwen3_5/model.h"

#include <algorithm>
#include <utility>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_5 {

backend::DeviceView Qwen35Executor::View(const std::string &name) const {
  const auto buffer = context_.buffers.find(name);
  if (buffer != context_.buffers.end()) {
    return buffer->second->view();
  }
  const auto weight = model_.weights().find(name);
  return weight == model_.weights().end() ? backend::DeviceView{} : weight->second.storage->view();
}

OperatorGraphBindings Qwen35Executor::Bindings() const {
  OperatorGraphBindings bindings =
      MakeOperatorGraphBindings([this](const std::string &name) { return View(name); });
  for (const auto &[name, weight] : model_.weights()) {
    bindings.values[name + ".type"] = static_cast<int32_t>(weight.type);
    bindings.values[name + ".scale"] = weight.scale;
  }
  return bindings;
}

bool Qwen35Executor::BindLayer(const int32_t layer, OperatorGraphBindings *const bindings) const {
  OMNI_RETURN_VAL_IF(bindings == nullptr || layer < 0 ||
                         static_cast<size_t>(layer) >= state_.layers.size(),
                     false);
  const bool is_recurrent = model_.model_config().IsRecurrent(layer);
  bindings->values["is_recurrent"] = is_recurrent;
  bindings->values["is_full_attention"] = !is_recurrent;
  const auto ffn_type =
      model_.weights().at("blk." + std::to_string(layer) + ".ffn_gate.weight").type;
  bindings->values["ffn_type"] = static_cast<int32_t>(ffn_type);
  const bool is_quantized_decode = std::get<bool>(bindings->values.at("is_quantized_decode"));
  bindings->values["ffn_block_size"] =
      ffn_type == util::GGUFTensorType::Q8_0 || is_quantized_decode ? 32 : 16;
  bindings->values["is_packed_nvfp4"] =
      ffn_type == util::GGUFTensorType::NVFP4 && !is_quantized_decode;
  bindings->values["is_packed_ffn_q8"] =
      ffn_type == util::GGUFTensorType::Q8_0 || is_quantized_decode;
  const auto &state = state_.layers[layer];
  if (is_recurrent) {
    bindings->values["convolution_state"] = state.convolution->view();
    bindings->values["recurrent_state"] = state.recurrent->view();
  } else {
    bindings->values["key_cache"] = state.key->view();
    bindings->values["value_cache"] = state.value->view();
  }
  return true;
}

bool Qwen35Executor::MakeLanguageNodes(const int32_t tokens, const int32_t start,
                                       const bool is_decode,
                                       std::vector<backend::ComputeNode> *const embedding,
                                       std::vector<backend::ComputeNode> *const nodes) {
  OMNI_RETURN_VAL_IF(nodes == nullptr || (!is_decode && embedding == nullptr) || tokens <= 0 ||
                         tokens > kPrefillBatchTokens,
                     SetError("invalid language graph output or batch size"));
  auto bindings = Bindings();
  const auto &c = model_.model_config();
  const int32_t recurrent = c.recurrent_value_heads * c.recurrent_head_size;
  const int32_t channels = recurrent + 2 * c.recurrent_key_heads * c.recurrent_head_size;
  const std::pair<std::string, OperatorGraphBindings::Value> values[] = {
      {"tokens", tokens},
      {"language_layers", c.layer_count},
      {"hidden_size", c.hidden_size},
      {"is_quantized_decode", tokens <= 8},
      {"ffn_size", c.feed_forward_size},
      {"query_heads", c.query_head_count},
      {"kv_heads", c.key_value_head_count},
      {"head_size", c.head_size},
      {"query_size", c.query_head_count * c.head_size},
      {"query_gate_size", 2 * c.query_head_count * c.head_size},
      {"kv_size", c.key_value_head_count * c.head_size},
      {"query_rows", tokens * c.query_head_count},
      {"kv_rows", tokens * c.key_value_head_count},
      {"recurrent_size", recurrent},
      {"recurrent_channels", channels},
      {"recurrent_rows", tokens * c.recurrent_value_heads},
      {"recurrent_heads", c.recurrent_value_heads},
      {"recurrent_groups", c.recurrent_value_heads / c.recurrent_key_heads},
      {"recurrent_key_heads", c.recurrent_key_heads},
      {"recurrent_head_size", c.recurrent_head_size},
      {"conv_size", c.convolution_size},
      {"epsilon", c.rms_norm_epsilon},
      {"rope_base", c.rope_frequency_base},
      {"rotary_size", c.rotary_size},
      {"rope_t", c.mrope_sections[0]},
      {"rope_h", c.mrope_sections[1]},
      {"rope_w", c.mrope_sections[2]},
      {"capacity", model_.config().max_context_tokens()},
      {"start_position", start},
      {"cached_tokens", is_decode ? model_.config().max_context_tokens() : start + tokens},
      {"vocabulary", c.vocabulary_size},
      {"hidden_values", static_cast<uint64_t>(tokens) * c.hidden_size},
      {"ffn_values", static_cast<uint64_t>(tokens) * c.feed_forward_size},
      {"query_values", static_cast<uint64_t>(tokens) * c.query_head_count * c.head_size},
      {"recurrent_values", static_cast<uint64_t>(tokens) * recurrent},
      {"token_input", View(is_decode ? "sample" : "tokens")},
      {"attention_position", is_decode ? View("cache_position") : backend::DeviceView{}},
      {"last_hidden",
       View("hidden").Slice(
           static_cast<size_t>(tokens - 1) * c.hidden_size * backend::ElementBytes(ComputeType()),
           static_cast<size_t>(c.hidden_size) * backend::ElementBytes(ComputeType()))},
  };
  for (const auto &[name, value] : values) {
    bindings.values[name] = value;
  }
  const OperatorIterationBinder bind_layer = [this](const std::string &name, const int32_t layer,
                                                    OperatorGraphBindings *const iteration) {
    return name != "layer" || BindLayer(layer, iteration);
  };
  nodes->clear();
  if (embedding != nullptr) {
    embedding->clear();
  }
  OMNI_RETURN_VAL_IF(
      !graph_config_.AppendSections({"embedding"}, bindings, bind_layer,
                                    is_decode ? nodes : embedding) ||
          !graph_config_.AppendSections({"language", "sampling"}, bindings, bind_layer, nodes),
      false);
  if (is_decode) {
    OMNI_RETURN_VAL_IF(!graph_config_.AppendSections({"decode"}, bindings, bind_layer, nodes),
                       false);
  }
  return true;
}

} // namespace qwen3_5
} // namespace inference
} // namespace omni_runtime
