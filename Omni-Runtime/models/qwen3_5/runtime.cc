#include "Omni-Runtime/models/qwen3_5/executor.h"
#include "Omni-Runtime/models/qwen3_5/model.h"

#include <algorithm>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_5 {

bool Qwen35Executor::Allocate(const std::string &name, const size_t bytes) {
  OMNI_RETURN_VAL_IF(context_.buffers.contains(name) || bytes == 0U,
                     SetError("invalid workspace allocation: " + name));
  memory::BufferPtr buffer;
  OMNI_RETURN_VAL_IF(
      !memory::AllocateDeviceBuffer(context_.backend, bytes, &context_.gpu_memory_bytes, &buffer),
      SetError("failed to allocate workspace: " + name));
  context_.buffers.emplace(name, std::move(buffer));
  return true;
}

bool Qwen35Executor::AllocateStorage() {
  const auto &c = model_.model_config();
  const auto &v = c.vision;
  const size_t batch = kPrefillBatchTokens;
  const size_t capacity = model_.config().max_context_tokens();
  const size_t query = static_cast<size_t>(c.query_head_count) * c.head_size;
  const size_t kv = static_cast<size_t>(c.key_value_head_count) * c.head_size;
  const size_t recurrent = static_cast<size_t>(c.recurrent_value_heads) * c.recurrent_head_size;
  const size_t channels =
      recurrent + 2U * static_cast<size_t>(c.recurrent_key_heads) * c.recurrent_head_size;
  const size_t patches = capacity * v.merge_size * v.merge_size;
  const size_t vision_hidden = patches * v.hidden_size;
  const size_t scalar_bytes = backend::ElementBytes(ComputeType());
  const size_t cache_bytes = c.has_fp8_weights ? scalar_bytes : sizeof(uint16_t);
  const std::pair<std::string, size_t> workspaces[] = {
      {"tokens", batch * sizeof(int32_t)},
      {"positions", batch * 3U * sizeof(int32_t)},
      {"cache_position", sizeof(int32_t)},
      {"sample", sizeof(int32_t)},
      {"hidden", batch * c.hidden_size * sizeof(float)},
      {"residual", batch * c.hidden_size * sizeof(float)},
      {"normalized", batch * c.hidden_size * sizeof(float)},
      {"normalized_q8", batch * c.hidden_size * sizeof(float)},
      {"normalized_q8_scales", batch * (c.hidden_size / 32) * sizeof(float)},
      {"normalized_ffn", batch * c.hidden_size * sizeof(float)},
      {"normalized_ffn_scales", batch * (c.hidden_size / 16) * sizeof(float)},
      {"projected", batch * c.hidden_size * sizeof(float)},
      {"qg", batch * query * 2U * sizeof(float)},
      {"query", batch * query * sizeof(float)},
      {"query_normed", batch * query * sizeof(float)},
      {"attention_gate", batch * query * sizeof(float)},
      {"key", batch * kv * sizeof(float)},
      {"key_normed", batch * kv * sizeof(float)},
      {"value", batch * kv * sizeof(float)},
      {"attention", batch * query * sizeof(float)},
      {"attention_quantized", batch * query * sizeof(float)},
      {"attention_scales", batch * (query / 32U) * sizeof(float)},
      {"gate", batch * c.feed_forward_size * sizeof(float)},
      {"up", batch * c.feed_forward_size * sizeof(float)},
      {"ffn_quantized", batch * c.feed_forward_size * sizeof(float)},
      {"ffn_scales", batch * (c.feed_forward_size / 16) * sizeof(float)},
      {"qkv", batch * channels * sizeof(float)},
      {"convolved", batch * channels * sizeof(float)},
      {"z", batch * recurrent * sizeof(float)},
      {"alpha", batch * c.recurrent_value_heads * sizeof(float)},
      {"beta", batch * c.recurrent_value_heads * sizeof(float)},
      {"delta", batch * recurrent * sizeof(float)},
      {"delta_normed", batch * recurrent * sizeof(float)},
      {"delta_quantized", batch * recurrent * sizeof(float)},
      {"delta_scales", batch * (recurrent / 32U) * sizeof(float)},
      {"output_quantized", static_cast<size_t>(c.hidden_size) * sizeof(float)},
      {"output_scales", static_cast<size_t>(c.hidden_size / 32) * sizeof(float)},
      {"logits", static_cast<size_t>(c.vocabulary_size) * sizeof(float)},
      {"decode_partials",
       ((capacity + 127U) / 128U) * c.query_head_count * (c.head_size + 2U) * sizeof(float)},
      {"vision_source", patches * v.patch_value_count()},
      {"vision_patches", patches * v.patch_value_count() * sizeof(float)},
      {"vision_patch_embedding", vision_hidden * sizeof(float)},
      {"vision_hidden", vision_hidden * sizeof(float)},
      {"vision_residual", vision_hidden * sizeof(float)},
      {"vision_normalized", vision_hidden * sizeof(float)},
      {"vision_qkv", vision_hidden * 3U * sizeof(float)},
      {"vision_query", vision_hidden * sizeof(float)},
      {"vision_key", vision_hidden * cache_bytes},
      {"vision_value", vision_hidden * cache_bytes},
      {"vision_attention", vision_hidden * sizeof(float)},
      {"vision_projected", vision_hidden * sizeof(float)},
      {"vision_feed_forward",
       patches * std::max(v.feed_forward_size, v.merged_hidden_size()) * sizeof(float)},
      {"image_embedding", capacity * c.hidden_size * sizeof(float)},
  };
  for (const auto &[name, bytes] : workspaces) {
    const bool is_fixed_width =
        name == "tokens" || name == "positions" || name == "cache_position" || name == "sample" ||
        name == "vision_source" || name == "vision_key" || name == "vision_value" ||
        name.ends_with("_scales") || name == "decode_partials";
    OMNI_RETURN_VAL_IF(
        !Allocate(name, is_fixed_width ? bytes : bytes / sizeof(float) * scalar_bytes), false);
  }
  state_.layers.resize(c.layer_count);
  for (int32_t layer = 0; layer < c.layer_count; ++layer) {
    auto &state = state_.layers[layer];
    if (c.IsRecurrent(layer)) {
      OMNI_RETURN_VAL_IF(
          !memory::AllocateDeviceBuffer(context_.backend,
                                        channels * (c.convolution_size - 1) * sizeof(float),
                                        &context_.gpu_memory_bytes, &state.convolution) ||
              !memory::AllocateDeviceBuffer(context_.backend,
                                            recurrent * c.recurrent_head_size * sizeof(float),
                                            &context_.gpu_memory_bytes, &state.recurrent),
          SetError("failed to allocate recurrent layer state"));
    } else {
      OMNI_RETURN_VAL_IF(
          !memory::AllocateDeviceBuffer(context_.backend, capacity * kv * cache_bytes,
                                        &context_.gpu_memory_bytes, &state.key) ||
              !memory::AllocateDeviceBuffer(context_.backend, capacity * kv * cache_bytes,
                                            &context_.gpu_memory_bytes, &state.value),
          SetError("failed to allocate attention KV cache"));
    }
  }
  return true;
}

bool Qwen35Executor::ResetState() {
  for (auto &state : state_.layers) {
    if (state.recurrent != nullptr) {
      OMNI_RETURN_VAL_IF(!state.recurrent->Fill(0U) || !state.convolution->Fill(0U),
                         SetError("failed to reset recurrent state"));
    }
  }
  OMNI_RETURN_VAL_IF(!context_.buffers.at("cache_position")->Fill(0U),
                     SetError("failed to reset decode cache position"));
  return true;
}

} // namespace qwen3_5
} // namespace inference
} // namespace omni_runtime
