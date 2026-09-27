#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_prefill_executor_gpu.h"

#include <algorithm>
#include <vector>

#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_language_executor_gpu.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_runtime.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {
namespace {

bool BuildMultiModalPrompt(const std::vector<int32_t> &prompt_token_ids,
                           const Qwen3VLConfig &model_config, const Qwen3VLVisionFeatures &features,
                           std::vector<int32_t> *const expanded_token_ids,
                           std::vector<int32_t> *const position_ids,
                           int32_t *const vision_token_offset) {
  OMNI_RETURN_VAL_IF(expanded_token_ids == nullptr || position_ids == nullptr ||
                         vision_token_offset == nullptr || features.token_count <= 0,
                     false);
  const auto image_iterator = std::find(prompt_token_ids.begin(), prompt_token_ids.end(),
                                        model_config.special_tokens.image_id);
  OMNI_RETURN_VAL_IF(image_iterator == prompt_token_ids.end(), false);
  *vision_token_offset =
      static_cast<int32_t>(std::distance(prompt_token_ids.begin(), image_iterator));
  expanded_token_ids->clear();
  expanded_token_ids->reserve(prompt_token_ids.size() - 1U + features.token_count);
  expanded_token_ids->insert(expanded_token_ids->end(), prompt_token_ids.begin(), image_iterator);
  expanded_token_ids->insert(expanded_token_ids->end(), features.token_count,
                             model_config.special_tokens.image_id);
  expanded_token_ids->insert(expanded_token_ids->end(), image_iterator + 1, prompt_token_ids.end());
  const int32_t token_count = static_cast<int32_t>(expanded_token_ids->size());
  position_ids->assign(static_cast<size_t>(token_count) * 3U, 0);
  int32_t text_position = 0;
  for (int32_t token = 0; token < *vision_token_offset; ++token) {
    for (int32_t axis = 0; axis < 3; ++axis) {
      (*position_ids)[static_cast<size_t>(axis) * token_count + token] = text_position;
    }
    ++text_position;
  }
  const int32_t vision_base = text_position;
  for (int32_t token = 0; token < features.token_count; ++token) {
    const int32_t output_token = *vision_token_offset + token;
    (*position_ids)[output_token] = vision_base;
    (*position_ids)[static_cast<size_t>(token_count) + output_token] =
        vision_base + token / features.merged_grid_width;
    (*position_ids)[static_cast<size_t>(token_count) * 2U + output_token] =
        vision_base + token % features.merged_grid_width;
  }
  text_position = vision_base + std::max(features.merged_grid_height * model_config.merge_size,
                                         features.merged_grid_width * model_config.merge_size) /
                                    model_config.merge_size;
  for (int32_t token = *vision_token_offset + features.token_count; token < token_count; ++token) {
    for (int32_t axis = 0; axis < 3; ++axis) {
      (*position_ids)[static_cast<size_t>(axis) * token_count + token] = text_position;
    }
    ++text_position;
  }
  return true;
}

} // namespace

Qwen3VLPrefillExecutorGpu::Qwen3VLPrefillExecutorGpu(Qwen3VLRuntime *const runtime,
                                                     Qwen3VLGenerationContext *const context)
    : runtime_(runtime), context_(context) {
}

bool Qwen3VLPrefillExecutorGpu::Initialize() {
  OMNI_RETURN_VAL_IF(runtime_ == nullptr, SetError("Qwen3-VL prefill runtime is null"));
  return true;
}

graph::Status Qwen3VLPrefillExecutorGpu::Run() {
  error_message_.clear();
  OMNI_RETURN_VAL_IF_LOG(runtime_ == nullptr || context_ == nullptr ||
                             context_->request == nullptr || context_->result == nullptr,
                         graph::InvalidArgumentStatus("Qwen3-VL prefill context is null"), ERROR,
                         "Qwen3-VL prefill context is null");
  const bool has_vision = context_->vision_features.embedding != nullptr;
  bool is_successful = false;
  if (has_vision) {
    is_successful =
        BuildMultiModalPrompt(context_->prompt_token_ids, runtime_->model_config,
                              context_->vision_features, &context_->prefill_token_ids,
                              &context_->prefill_position_ids, &context_->vision_token_offset);
  } else {
    context_->prefill_token_ids = context_->prompt_token_ids;
    context_->prefill_position_ids.clear();
    context_->vision_token_offset = -1;
    is_successful = true;
  }
  OMNI_RETURN_VAL_IF_LOG(
      context_->prefill_token_ids.size() + context_->request->max_new_tokens >
          static_cast<size_t>(runtime_->cache_capacity),
      graph::InvalidArgumentStatus("Qwen3-VL prompt and generation exceed context capacity"), ERROR,
      "Qwen3-VL prompt and generation exceed context capacity");
  if (is_successful) {
    is_successful = RunQwen3VLLanguageStep(
        runtime_, context_->prefill_token_ids, context_->prefill_position_ids,
        context_->vision_features, context_->vision_token_offset, &context_->next_token_id);
  }
  OMNI_RETURN_VAL_IF_LOG(
      !is_successful,
      graph::InternalStatus(error_message_.empty() ? "Qwen3-VL prefill failed" : error_message_),
      ERROR, error_message_);
  context_->result->backend_name =
      std::string("native_") + backend::BackendName(runtime_->backend->kind()) + "(omni-runtime)";
  context_->result->prefill_token_count = static_cast<int32_t>(context_->prefill_token_ids.size());
  context_->result->vision_token_count = context_->vision_features.token_count;
  context_->result->prompt_token_ids = context_->prefill_token_ids;
  context_->result->prefill_time_ms = runtime_->step_time_ms;
  context_->result->gpu_memory_bytes = runtime_->gpu_memory_bytes;
  return graph::Status();
}

bool Qwen3VLPrefillExecutorGpu::SetError(const std::string &message) {
  error_message_ = message;
  if (context_ != nullptr) {
    context_->error_message = message;
  }
  return false;
}

const std::string &Qwen3VLPrefillExecutorGpu::error_message() const {
  return error_message_;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
