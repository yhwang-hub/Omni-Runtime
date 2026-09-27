#include "Omni-Runtime/models/qwen3_8_nvfp4/qwen3_8_nvfp4_pipeline.h"

#include <algorithm>
#include <utility>

#include "Omni-Runtime/inference/common/image_preprocessor/image_preprocessor.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime::inference::qwen3_8_nvfp4 {

backend::DeviceView Qwen38Pipeline::View(const std::string &name) const {
  const auto buffer = buffers_.find(name);
  if (buffer != buffers_.end()) {
    return buffer->second->view();
  }
  const auto weight = weights_.find(name);
  return weight == weights_.end() ? backend::DeviceView{} : weight->second.storage->view();
}

OperatorGraphBindings Qwen38Pipeline::Bindings() const {
  OperatorGraphBindings bindings;
  bindings.resolve_view = [this](const std::string &name, backend::DeviceView *const view) {
    OMNI_RETURN_VAL_IF(view == nullptr, false);
    *view = View(name);
    return view->IsValid();
  };
  for (const auto &[name, weight] : weights_) {
    bindings.values[name + ".type"] = weight.type;
    bindings.values[name + ".scale"] = weight.scale;
  }
  return bindings;
}

bool Qwen38Pipeline::BindLayer(const int32_t layer, OperatorGraphBindings *const bindings) const {
  OMNI_RETURN_VAL_IF(
      bindings == nullptr || layer < 0 || static_cast<size_t>(layer) >= states_.size(), false);
  const bool is_recurrent = model_config_.IsRecurrent(layer);
  bindings->values["is_recurrent"] = is_recurrent;
  bindings->values["is_full_attention"] = !is_recurrent;
  const int32_t ffn_type = weights_.at("blk." + std::to_string(layer) + ".ffn_gate.weight").type;
  bindings->values["ffn_type"] = ffn_type;
  const bool is_quantized_decode = std::get<bool>(bindings->values.at("is_quantized_decode"));
  bindings->values["ffn_block_size"] = ffn_type == 8 || is_quantized_decode ? 32 : 16;
  bindings->values["is_packed_nvfp4"] = ffn_type == 40 && !is_quantized_decode;
  bindings->values["is_packed_ffn_q8"] = ffn_type == 8 || is_quantized_decode;
  const auto &state = states_[layer];
  if (is_recurrent) {
    bindings->values["convolution_state"] = state.convolution->view();
    bindings->values["recurrent_state"] = state.recurrent->view();
  } else {
    bindings->values["key_cache"] = state.key->view();
    bindings->values["value_cache"] = state.value->view();
  }
  return true;
}

bool Qwen38Pipeline::MakeLanguageNodes(const int32_t tokens, const int32_t start,
                                       const bool is_decode,
                                       std::vector<backend::ComputeNode> *const embedding,
                                       std::vector<backend::ComputeNode> *const nodes) {
  OMNI_RETURN_VAL_IF(nodes == nullptr || (!is_decode && embedding == nullptr) || tokens <= 0 ||
                         tokens > kPrefillBatchTokens,
                     SetError("invalid language graph output or batch size"));
  auto bindings = Bindings();
  const auto &c = model_config_;
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
      {"recurrent_key_heads", c.recurrent_key_heads},
      {"recurrent_head_size", c.recurrent_head_size},
      {"conv_size", c.convolution_size},
      {"epsilon", c.rms_norm_epsilon},
      {"rope_base", c.rope_frequency_base},
      {"rotary_size", c.rotary_size},
      {"rope_t", c.mrope_sections[0]},
      {"rope_h", c.mrope_sections[1]},
      {"rope_w", c.mrope_sections[2]},
      {"capacity", inference_config_.max_context_tokens()},
      {"start_position", start},
      {"cached_tokens", is_decode ? inference_config_.max_context_tokens() : start + tokens},
      {"vocabulary", c.vocabulary_size},
      {"hidden_values", static_cast<uint64_t>(tokens) * c.hidden_size},
      {"ffn_values", static_cast<uint64_t>(tokens) * c.feed_forward_size},
      {"query_values", static_cast<uint64_t>(tokens) * c.query_head_count * c.head_size},
      {"recurrent_values", static_cast<uint64_t>(tokens) * recurrent},
      {"token_input", View(is_decode ? "sample" : "tokens")},
      {"attention_position", is_decode ? View("cache_position") : backend::DeviceView{}},
      {"last_hidden",
       View("hidden").Slice(static_cast<size_t>(tokens - 1) * c.hidden_size * sizeof(float),
                            static_cast<size_t>(c.hidden_size) * sizeof(float))},
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
      !graph_config_.Append("embedding", bindings, bind_layer, is_decode ? nodes : embedding),
      SetError(graph_config_.error_message()));
  for (const std::string section : {"language", "sampling"}) {
    OMNI_RETURN_VAL_IF(!graph_config_.Append(section, bindings, bind_layer, nodes),
                       SetError(graph_config_.error_message()));
  }
  if (is_decode) {
    OMNI_RETURN_VAL_IF(!graph_config_.Append("decode", bindings, bind_layer, nodes),
                       SetError(graph_config_.error_message()));
  }
  return true;
}

bool Qwen38Pipeline::PrepareDecode() {
  std::vector<backend::ComputeNode> nodes;
  OMNI_RETURN_VAL_IF(!MakeLanguageNodes(1, 0, true, nullptr, &nodes), false);
  OMNI_RETURN_VAL_IF(!backend_->CreateExecutionPlan(backend::ScalarType::kFloat32, nodes,
                                                    buffers_.at("sample")->allocation(),
                                                    &decode_plan_),
                     SetError("failed to prepare resident decode graph"));
  return true;
}

bool Qwen38Pipeline::Prefill(const Prompt &prompt, int32_t *const next_token,
                             GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(next_token == nullptr || result == nullptr,
                     SetError("prefill output is null"));
  const int32_t total = static_cast<int32_t>(prompt.tokens.size());
  const auto &c = model_config_;
  for (int32_t first = 0; first < total;) {
    int32_t segment_end = total;
    if (prompt.image_offset > first) {
      segment_end = prompt.image_offset;
    } else if (first < prompt.image_offset + prompt.image_tokens) {
      segment_end = prompt.image_offset + prompt.image_tokens;
    }
    const int32_t count = std::min(kPrefillBatchTokens, segment_end - first);
    OMNI_RETURN_VAL_IF(count <= 0, SetError("invalid multimodal prefill segment"));
    std::vector<backend::ComputeNode> nodes;
    std::vector<backend::ComputeNode> embedding;
    OMNI_RETURN_VAL_IF(!MakeLanguageNodes(count, first, false, &embedding, &nodes), false);
    std::unique_ptr<backend::ExecutionPlan> plan;
    OMNI_RETURN_VAL_IF(!backend_->CreateExecutionPlan(backend::ScalarType::kFloat32, nodes,
                                                      buffers_.at("sample")->allocation(), &plan),
                       SetError("failed to prepare prefill graph"));
    std::vector<int32_t> positions(static_cast<size_t>(count) * 3U);
    for (int32_t axis = 0; axis < 3; ++axis) {
      std::copy_n(prompt.positions.begin() + static_cast<size_t>(axis) * total + first, count,
                  positions.begin() + static_cast<size_t>(axis) * count);
    }
    OMNI_RETURN_VAL_IF(!timer_->Start() ||
                           !buffers_.at("tokens")->CopyFromHost(prompt.tokens.data() + first,
                                                                count * sizeof(int32_t)) ||
                           !buffers_.at("positions")->CopyDataFromVector(positions),
                       SetError("failed to execute token embedding"));
    for (const auto &node : embedding) {
      OMNI_RETURN_VAL_IF(!backend_->Compute(backend::ScalarType::kFloat32, node.operation),
                         SetError("failed to execute token embedding"));
    }
    if (prompt.image_offset >= 0 && first >= prompt.image_offset &&
        first < prompt.image_offset + prompt.image_tokens) {
      const size_t bytes = static_cast<size_t>(count) * c.hidden_size * sizeof(float);
      const auto source = memory::Buffer::CreateDeviceBufferView(
          buffers_.at("image_embedding"),
          static_cast<size_t>(first - prompt.image_offset) * c.hidden_size * sizeof(float), bytes);
      const auto destination =
          memory::Buffer::CreateDeviceBufferView(buffers_.at("hidden"), 0U, bytes);
      OMNI_RETURN_VAL_IF(source == nullptr || destination == nullptr ||
                             !destination->CopyFromBuffer(*source),
                         SetError("failed to inject vision embeddings"));
    }
    double graph_ms = 0.0;
    double elapsed_ms = 0.0;
    OMNI_RETURN_VAL_IF(!plan->Run(sizeof(*next_token), next_token, &graph_ms) ||
                           !timer_->Stop(&elapsed_ms),
                       SetError("prefill execution failed"));
    result->prefill_time_ms += elapsed_ms;
    first += count;
  }
  const int32_t positions[] = {prompt.next_position, prompt.next_position, prompt.next_position};
  OMNI_RETURN_VAL_IF(!buffers_.at("positions")->CopyFromHost(positions, sizeof(positions)) ||
                         !buffers_.at("cache_position")->CopyFromHost(&total, sizeof(total)),
                     SetError("failed to initialize decode positions"));
  result->backend_name = "native_" + compute_backend() + "(omni-runtime)";
  result->prefill_token_count = total;
  result->prompt_token_ids = prompt.tokens;
  result->vision_token_count = prompt.image_tokens;
  result->gpu_memory_bytes = gpu_memory_bytes_ + backend_->compute_workspace_bytes();
  return true;
}

bool Qwen38Pipeline::EncodeImage(const GenerationRequest &request, Prompt *const prompt,
                                 GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(prompt == nullptr || result == nullptr, SetError("vision output is null"));
  if (request.image_path.empty()) {
    return true;
  }
  const auto &v = model_config_.vision;
  std::vector<uint8_t> pixels;
  int32_t width_pixels = 0;
  int32_t height_pixels = 0;
  ImagePatchGrid grid;
  OMNI_RETURN_VAL_IF(!ImagePreprocessor::LoadImage(request.image_path, true, &pixels, &width_pixels,
                                                   &height_pixels) ||
                         !ImagePreprocessor::MakeGrid(width_pixels, height_pixels, v.patch_size,
                                                      v.merge_size, &grid),
                     SetError("cannot prepare input image"));
  OMNI_RETURN_VAL_IF(grid.merged_token_count > inference_config_.max_context_tokens() ||
                         pixels.size() > buffers_.at("vision_source")->size_bytes(),
                     SetError("input image exceeds allocated context storage"));
  auto bindings = Bindings();
  const std::pair<std::string, OperatorGraphBindings::Value> values[] = {
      {"vision_layers", v.layer_count},
      {"v_tokens", grid.token_count},
      {"v_hidden", v.hidden_size},
      {"v_ffn", v.feed_forward_size},
      {"v_qkv", 3 * v.hidden_size},
      {"v_heads", v.head_count},
      {"v_head_size", v.head_size()},
      {"v_epsilon", v.layer_norm_epsilon},
      {"grid_height", grid.height_patches},
      {"grid_width", grid.width_patches},
      {"patch_size", v.patch_size},
      {"merge_size", v.merge_size},
      {"position_grid_size", v.image_size / v.patch_size},
      {"resized_width", grid.width_patches * v.patch_size},
      {"resized_height", grid.height_patches * v.patch_size},
      {"input_width", width_pixels},
      {"input_height", height_pixels},
      {"pixels", static_cast<uint64_t>(grid.width_patches) * grid.height_patches * v.patch_size *
                     v.patch_size},
      {"v_hidden_values", static_cast<uint64_t>(grid.token_count) * v.hidden_size},
      {"v_qkv_values", static_cast<uint64_t>(grid.token_count) * v.hidden_size * 3U},
      {"v_ffn_values", static_cast<uint64_t>(grid.token_count) * v.feed_forward_size},
      {"merged_tokens", grid.merged_token_count},
      {"merged_hidden", v.merged_hidden_size()},
      {"merged_values", static_cast<uint64_t>(grid.merged_token_count) * v.merged_hidden_size()},
      {"projection_size", v.projection_size},
      {"embedding_values", static_cast<uint64_t>(grid.merged_token_count) * v.projection_size},
  };
  for (const auto &[name, value] : values) {
    bindings.values[name] = value;
  }
  std::vector<backend::ComputeNode> nodes;
  OMNI_RETURN_VAL_IF(!graph_config_.Append("vision", bindings, {}, &nodes),
                     SetError(graph_config_.error_message()));
  const auto readback =
      memory::Buffer::CreateDeviceBufferView(buffers_.at("image_embedding"), 0U, sizeof(float));
  std::unique_ptr<backend::ExecutionPlan> plan;
  OMNI_RETURN_VAL_IF(readback == nullptr ||
                         !backend_->CreateExecutionPlan(backend::ScalarType::kFloat32, nodes,
                                                        readback->allocation(), &plan),
                     SetError("failed to prepare vision execution graph"));
  float first_embedding = 0.0F;
  double graph_time_ms = 0.0;
  OMNI_RETURN_VAL_IF(!timer_->Start() ||
                         !buffers_.at("vision_source")->CopyDataFromVector(pixels) ||
                         !plan->Run(sizeof(first_embedding), &first_embedding, &graph_time_ms) ||
                         !timer_->Stop(&result->vision_time_ms),
                     SetError("vision execution failed"));
  prompt->image_tokens = grid.merged_token_count;
  prompt->image_height = grid.height_patches / v.merge_size;
  prompt->image_width = grid.width_patches / v.merge_size;
  result->input_image_size = {width_pixels, height_pixels};
  result->resized_image_size = {grid.width_patches * v.patch_size,
                                grid.height_patches * v.patch_size};
  return true;
}

} // namespace omni_runtime::inference::qwen3_8_nvfp4
