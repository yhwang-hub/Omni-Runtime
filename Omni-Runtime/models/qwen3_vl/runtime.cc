#include "Omni-Runtime/models/qwen3_vl/executor.h"
#include "Omni-Runtime/models/qwen3_vl/model.h"

#include <algorithm>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

size_t Qwen3VLExecutor::RuntimeElementSize() const {
  switch (model_.data().config.compute_type()) {
  case ModelWeightType::kFloat32:
    return sizeof(float);
  case ModelWeightType::kFloat16:
  case ModelWeightType::kBFloat16:
    return sizeof(uint16_t);
  case ModelWeightType::kInt8:
    return sizeof(uint8_t);
  case ModelWeightType::kAuto:
    break;
  }
  return sizeof(float);
}

bool Qwen3VLExecutor::AllocateKvCache(const int32_t capacity) {
  OMNI_RETURN_VAL_IF_LOG(context_ == nullptr || capacity <= 0, false, ERROR,
                         "Invalid KV allocation");
  const int32_t key_value_size =
      model_.data().model_config.key_value_head_count * model_.data().model_config.head_size;
  const uint64_t element_count = static_cast<uint64_t>(model_.data().model_config.layer_count) *
                                 static_cast<uint64_t>(capacity) *
                                 static_cast<uint64_t>(key_value_size);
  const size_t bytes = static_cast<size_t>(element_count) * RuntimeElementSize();
  OMNI_RETURN_VAL_IF(
      !memory::AllocateDeviceBuffer(model_.data().backend, bytes, &context_->gpu_memory_bytes,
                                    &state_.key_cache) ||
          !memory::AllocateDeviceBuffer(model_.data().backend, bytes, &context_->gpu_memory_bytes,
                                        &state_.value_cache),
      false);
  state_.cache_capacity = capacity;
  return true;
}

bool Qwen3VLExecutor::ResetState() {
  state_.used_token_count = 0;
  state_.next_rope_position = 0;
  return true;
}

bool Qwen3VLExecutor::AllocateWorkspaces() {
  OMNI_RETURN_VAL_IF_LOG(context_ == nullptr, false, ERROR, "Qwen3-VL execution context is null");
  const size_t count = model_.data().config.max_context_tokens();
  const size_t element_size = RuntimeElementSize();
  const Qwen3VLConfig &language = model_.data().model_config;
  const qwen3::Qwen3VisionConfig &vision = model_.data().vision_config;
  LanguageWorkspaces &l = context_->language_workspaces;
  VisionWorkspaces &v = context_->vision_workspaces;
  const size_t vision_count = count * vision.merge_size * vision.merge_size;
  const size_t hidden_count = vision_count * vision.hidden_size;
  const std::pair<memory::BufferPtr *, size_t> buffers[] = {
      {&state_.decode_cache_position, sizeof(int32_t)},
      {&l.tokens, count * sizeof(int32_t)},
      {&l.positions, count * 3U * sizeof(int32_t)},
      {&l.hidden, count * language.hidden_size * element_size},
      {&l.next_hidden, count * language.hidden_size * element_size},
      {&l.normalized, count * language.hidden_size * element_size},
      {&l.query, count * language.query_head_count * language.head_size * element_size},
      {&l.key, count * language.key_value_head_count * language.head_size * element_size},
      {&l.value, count * language.key_value_head_count * language.head_size * element_size},
      {&l.query_normed, count * language.query_head_count * language.head_size * element_size},
      {&l.key_normed, count * language.key_value_head_count * language.head_size * element_size},
      {&l.attention, count * language.query_head_count * language.head_size * element_size},
      {&l.projected, count * language.hidden_size * element_size},
      {&l.gate, count * language.feed_forward_size * element_size},
      {&l.up, count * language.feed_forward_size * element_size},
      {&l.logits, static_cast<size_t>(language.vocabulary_size) * element_size},
      {&l.sample, sizeof(int32_t)},
      {&l.decode_partials, ((count + 127U) / 128U) * language.query_head_count *
                               (language.head_size + 2U) * sizeof(float)},
      {&v.source, vision_count * vision.patch_value_count()},
      {&v.patches, vision_count * vision.patch_value_count() * element_size},
      {&v.patch_embedding, hidden_count * element_size},
      {&v.hidden, hidden_count * element_size},
      {&v.residual, hidden_count * element_size},
      {&v.normalized, hidden_count * element_size},
      {&v.qkv, hidden_count * 3U * element_size},
      {&v.query, hidden_count * element_size},
      {&v.key, hidden_count * element_size},
      {&v.value, hidden_count * element_size},
      {&v.attention, hidden_count * element_size},
      {&v.projected, hidden_count * element_size},
      {&v.feed_forward, vision_count *
                            std::max(vision.feed_forward_size, vision.merged_hidden_size()) *
                            element_size},
      {&v.embedding, count * vision.projection_size * element_size},
  };
  for (const auto &entry : buffers) {
    OMNI_RETURN_VAL_IF(!memory::AllocateDeviceBuffer(model_.data().backend, entry.second,
                                                     &context_->gpu_memory_bytes, entry.first),
                       false);
  }
  v.deepstack.resize(vision.deepstack_layers.size());
  for (memory::BufferPtr &buffer : v.deepstack) {
    OMNI_RETURN_VAL_IF(!memory::AllocateDeviceBuffer(model_.data().backend,
                                                     count * vision.projection_size * element_size,
                                                     &context_->gpu_memory_bytes, &buffer),
                       false);
  }
  return true;
}

backend::ScalarType Qwen3VLExecutor::RuntimeScalarType() const {
  switch (model_.data().config.compute_type()) {
  case ModelWeightType::kFloat16:
    return backend::ScalarType::kFloat16;
  case ModelWeightType::kBFloat16:
    return backend::ScalarType::kBFloat16;
  default:
    return backend::ScalarType::kFloat32;
  }
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
