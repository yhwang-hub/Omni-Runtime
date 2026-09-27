#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_language_executor_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Omni-Runtime/inference/tokenizer/tokenizer.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_runtime.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {
namespace {

using memory::BufferPtr;

using LanguageWorkspaces = Qwen3VLLanguageWorkspaces;

bool LaunchEmbedding(Qwen3VLRuntime *const runtime, const void *const tokens,
                     const int32_t token_count, void *const output) {
  const Qwen3VLConfig &config = runtime->model_config;
  return runtime->backend->Compute(
      RuntimeScalarType(*runtime),
      backend::Embedding{.indices = tokens,
                         .weight = Weight(*runtime, "token_embd.weight"),
                         .token_count = token_count,
                         .hidden_size = config.hidden_size,
                         .output = output});
}

bool LaunchRmsNorm(Qwen3VLRuntime *const runtime, const void *const input,
                   const std::string &weight_name, const int32_t rows, const int32_t hidden_size,
                   void *const output) {
  return runtime->backend->Compute(
      RuntimeScalarType(*runtime),
      backend::RmsNorm{.input = input,
                       .weight = Weight(*runtime, weight_name),
                       .rows = rows,
                       .hidden_size = hidden_size,
                       .epsilon = runtime->model_config.rms_norm_epsilon,
                       .output = output});
}

bool LaunchLayer(Qwen3VLRuntime *const runtime, const int32_t layer, const int32_t token_count,
                 LanguageWorkspaces *const workspaces) {
  OMNI_RETURN_VAL_IF(runtime == nullptr || workspaces == nullptr, false);
  const Qwen3VLConfig &config = runtime->model_config;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  const int32_t query_size = config.query_head_count * config.head_size;
  const int32_t key_value_size = config.key_value_head_count * config.head_size;
  const int32_t start_position = runtime->used_token_count;
  const uint64_t hidden_values = static_cast<uint64_t>(token_count) * config.hidden_size;
  const int32_t kv_offset = layer * runtime->cache_capacity * key_value_size;
  OMNI_RETURN_VAL_IF(
      !LaunchRmsNorm(runtime, workspaces->hidden->device_address(), prefix + "attn_norm.weight",
                     token_count, config.hidden_size, workspaces->normalized->device_address()) ||
          !LaunchMatmul(runtime, workspaces->normalized->device_address(), prefix + "attn_q.weight",
                        token_count, config.hidden_size, query_size,
                        workspaces->query->device_address()) ||
          !LaunchMatmul(runtime, workspaces->normalized->device_address(), prefix + "attn_k.weight",
                        token_count, config.hidden_size, key_value_size,
                        workspaces->key->device_address()) ||
          !LaunchMatmul(runtime, workspaces->normalized->device_address(), prefix + "attn_v.weight",
                        token_count, config.hidden_size, key_value_size,
                        workspaces->value->device_address()) ||
          !LaunchRmsNorm(runtime, workspaces->query->device_address(),
                         prefix + "attn_q_norm.weight", token_count * config.query_head_count,
                         config.head_size, workspaces->query_normed->device_address()) ||
          !LaunchRmsNorm(runtime, workspaces->key->device_address(), prefix + "attn_k_norm.weight",
                         token_count * config.key_value_head_count, config.head_size,
                         workspaces->key_normed->device_address()) ||
          !runtime->backend->Compute(
              RuntimeScalarType(*runtime),
              backend::MultiAxisRope{.query = workspaces->query_normed->device_address(),
                                     .key = workspaces->key_normed->device_address(),
                                     .position_ids = workspaces->positions->device_address(),
                                     .tokens = token_count,
                                     .query_heads = config.query_head_count,
                                     .key_value_heads = config.key_value_head_count,
                                     .head_dim = config.head_size,
                                     .temporal_section = config.mrope_sections[0],
                                     .height_section = config.mrope_sections[1],
                                     .width_section = config.mrope_sections[2],
                                     .frequency_base = config.rope_frequency_base,
                                     .query_output = workspaces->query->device_address(),
                                     .key_output = workspaces->key->device_address()}),
      false);
  OMNI_RETURN_VAL_IF(
      !runtime->backend->Compute(
          RuntimeScalarType(*runtime),
          backend::StoreKvCache{
              .key = workspaces->key->device_address(),
              .value = workspaces->value->device_address(),
              .token_count = token_count,
              .key_value_size = key_value_size,
              .start_position = start_position,
              .cache_capacity = runtime->cache_capacity,
              .key_cache = static_cast<std::uint8_t *>(runtime->key_cache->device_address()) +
                           static_cast<size_t>(kv_offset) * RuntimeElementSize(*runtime),
              .value_cache = static_cast<std::uint8_t *>(runtime->value_cache->device_address()) +
                             static_cast<size_t>(kv_offset) * RuntimeElementSize(*runtime)}),
      false);
  OMNI_RETURN_VAL_IF(
      !runtime->backend->Compute(
          RuntimeScalarType(*runtime),
          backend::Attention{
              .query = workspaces->query->device_address(),
              .key_cache = static_cast<std::uint8_t *>(runtime->key_cache->device_address()) +
                           static_cast<size_t>(kv_offset) * RuntimeElementSize(*runtime),
              .value_cache = static_cast<std::uint8_t *>(runtime->value_cache->device_address()) +
                             static_cast<size_t>(kv_offset) * RuntimeElementSize(*runtime),
              .query_token_count = token_count,
              .query_head_count = config.query_head_count,
              .key_value_head_count = config.key_value_head_count,
              .cached_token_count = start_position + token_count,
              .start_position = start_position,
              .head_dim = config.head_size,
              .is_causal = true,
              .output = workspaces->attention->device_address(),
              .decode_partials = workspaces->decode_partials->device_address()}) ||
          !LaunchMatmul(runtime, workspaces->attention->device_address(),
                        prefix + "attn_output.weight", token_count, query_size, config.hidden_size,
                        workspaces->projected->device_address()) ||
          !runtime->backend->Compute(
              RuntimeScalarType(*runtime),
              backend::Add{.lhs = workspaces->projected->device_address(),
                           .rhs = workspaces->hidden->device_address(),
                           .element_count = hidden_values,
                           .output = workspaces->next_hidden->device_address()}) ||
          !LaunchRmsNorm(runtime, workspaces->next_hidden->device_address(),
                         prefix + "ffn_norm.weight", token_count, config.hidden_size,
                         workspaces->normalized->device_address()) ||
          !LaunchMatmul(runtime, workspaces->normalized->device_address(),
                        prefix + "ffn_gate.weight", token_count, config.hidden_size,
                        config.feed_forward_size, workspaces->gate->device_address()) ||
          !LaunchMatmul(runtime, workspaces->normalized->device_address(), prefix + "ffn_up.weight",
                        token_count, config.hidden_size, config.feed_forward_size,
                        workspaces->up->device_address()) ||
          !runtime->backend->Compute(
              RuntimeScalarType(*runtime),
              backend::SwiGLU{.gate = workspaces->gate->device_address(),
                              .up = workspaces->up->device_address(),
                              .element_count =
                                  static_cast<uint64_t>(token_count) * config.feed_forward_size,
                              .output = workspaces->gate->device_address()}) ||
          !LaunchMatmul(runtime, workspaces->gate->device_address(), prefix + "ffn_down.weight",
                        token_count, config.feed_forward_size, config.hidden_size,
                        workspaces->projected->device_address()) ||
          !runtime->backend->Compute(RuntimeScalarType(*runtime),
                                     backend::Add{.lhs = workspaces->projected->device_address(),
                                                  .rhs = workspaces->next_hidden->device_address(),
                                                  .element_count = hidden_values,
                                                  .output = workspaces->hidden->device_address()}),
      false);
  return true;
}

bool LaunchOutput(Qwen3VLRuntime *const runtime, const int32_t token_count,
                  LanguageWorkspaces *const workspaces) {
  const Qwen3VLConfig &config = runtime->model_config;
  const int32_t last_row = (token_count - 1) * config.hidden_size;
  if (!LaunchRmsNorm(runtime,
                     static_cast<const std::uint8_t *>(workspaces->hidden->device_address()) +
                         static_cast<size_t>(last_row) * RuntimeElementSize(*runtime),
                     "output_norm.weight", 1, config.hidden_size,
                     workspaces->normalized->device_address()) ||
      !LaunchMatmul(runtime, workspaces->normalized->device_address(),
                    runtime->registry->has_output_weight() ? "output.weight" : "token_embd.weight",
                    1, config.hidden_size, config.vocabulary_size,
                    workspaces->logits->device_address())) {
    return false;
  }
  return runtime->backend->Compute(RuntimeScalarType(*runtime),
                                   backend::Argmax{.logits = workspaces->logits->device_address(),
                                                   .rows = 1,
                                                   .columns = config.vocabulary_size,
                                                   .output = workspaces->sample->device_address()});
}

bool RunLanguageStep(Qwen3VLRuntime *const runtime, const std::vector<int32_t> &token_ids,
                     const std::vector<int32_t> &input_position_ids,
                     const Qwen3VLVisionFeatures &features, const int32_t vision_token_offset,
                     int32_t *const next_token_id) {
  OMNI_RETURN_VAL_IF(runtime == nullptr || next_token_id == nullptr || token_ids.empty(), false);
  const int32_t token_count = static_cast<int32_t>(token_ids.size());
  const bool is_decode = token_count == 1 && features.embedding == nullptr;
  OMNI_RETURN_VAL_IF_LOG(runtime->used_token_count + token_count > runtime->cache_capacity, false,
                         ERROR, "Qwen3-VL context capacity exceeded");
  std::vector<int32_t> position_ids = input_position_ids;
  if (position_ids.empty()) {
    position_ids.resize(static_cast<size_t>(token_count) * 3U, 0);
    for (int32_t token = 0; token < token_count; ++token) {
      const int32_t position = runtime->used_token_count + token;
      for (int32_t axis = 0; axis < 3; ++axis) {
        position_ids[static_cast<size_t>(axis) * token_count + token] = position;
      }
    }
  }
  LanguageWorkspaces &workspaces = runtime->language_workspaces;
  OMNI_RETURN_VAL_IF(!runtime->compute_timer.Start(), false);
  OMNI_RETURN_VAL_IF(
      !workspaces.tokens->CopyFromHost(token_ids.data(), token_count * sizeof(int32_t)) ||
          !workspaces.positions->CopyFromHost(position_ids.data(),
                                              position_ids.size() * sizeof(int32_t)),
      false);
  OMNI_RETURN_VAL_IF(!LaunchEmbedding(runtime, workspaces.tokens->device_address(), token_count,
                                      workspaces.hidden->device_address()),
                     false);
  if (features.embedding != nullptr) {
    const size_t offset = static_cast<size_t>(vision_token_offset) *
                          runtime->model_config.hidden_size * RuntimeElementSize(*runtime);
    const size_t bytes = static_cast<size_t>(features.token_count) *
                         runtime->model_config.hidden_size * RuntimeElementSize(*runtime);
    const memory::BufferPtr destination = memory::Buffer::CreateDeviceBufferView(
        runtime->backend, static_cast<std::uint8_t *>(workspaces.hidden->device_address()) + offset,
        bytes);
    OMNI_RETURN_VAL_IF(destination == nullptr || !runtime->backend->CopyDeviceToDevice(
                                                     features.embedding->allocation(),
                                                     destination->allocation(), bytes, nullptr),
                       false);
  }
  for (int32_t layer = 0; layer < runtime->model_config.layer_count; ++layer) {
    OMNI_RETURN_VAL_IF(!LaunchLayer(runtime, layer, token_count, &workspaces), false);
    if (layer >= static_cast<int32_t>(features.deepstack_embeddings.size())) {
      continue;
    }
    const size_t offset = static_cast<size_t>(vision_token_offset) *
                          runtime->model_config.hidden_size * RuntimeElementSize(*runtime);
    const size_t bytes = static_cast<size_t>(features.token_count) *
                         runtime->model_config.hidden_size * RuntimeElementSize(*runtime);
    const memory::BufferPtr source = features.deepstack_embeddings[layer];
    const memory::BufferPtr destination = memory::Buffer::CreateDeviceBufferView(
        runtime->backend, static_cast<std::uint8_t *>(workspaces.hidden->device_address()) + offset,
        bytes);
    OMNI_RETURN_VAL_IF(
        destination == nullptr ||
            !runtime->backend->Compute(
                RuntimeScalarType(*runtime),
                backend::Add{.lhs = destination->device_address(),
                             .rhs = source->device_address(),
                             .element_count = static_cast<uint64_t>(features.token_count) *
                                              runtime->model_config.hidden_size,
                             .output = destination->device_address()}),
        false);
  }
  OMNI_RETURN_VAL_IF(!LaunchOutput(runtime, token_count, &workspaces), false);
  OMNI_RETURN_VAL_IF(!runtime->compute_timer.Stop(&runtime->step_time_ms), false);
  int32_t sampled_token = 0;
  OMNI_RETURN_VAL_IF(!workspaces.sample->CopyToHost(sizeof(sampled_token), &sampled_token), false);
  *next_token_id = sampled_token;
  runtime->used_token_count += token_count;
  if (is_decode) {
    ++runtime->next_rope_position;
  } else {
    runtime->next_rope_position =
        position_ids.empty() ? token_count
                             : *std::max_element(position_ids.begin(), position_ids.end()) + 1;
  }
  return true;
}

} // namespace

bool RunQwen3VLLanguageStep(Qwen3VLRuntime *const runtime, const std::vector<int32_t> &token_ids,
                            const std::vector<int32_t> &position_ids,
                            const Qwen3VLVisionFeatures &vision_features,
                            const int32_t vision_token_offset, int32_t *const next_token_id) {
  OMNI_RETURN_VAL_IF(runtime == nullptr, false);
  return RunLanguageStep(runtime, token_ids, position_ids, vision_features, vision_token_offset,
                         next_token_id);
}
} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
