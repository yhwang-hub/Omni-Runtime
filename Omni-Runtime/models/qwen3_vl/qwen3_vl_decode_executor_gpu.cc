#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_decode_executor_gpu.h"

#include <vector>

#include "Omni-Runtime/inference/common/text_stream/text_stream.h"
#include "Omni-Runtime/inference/tokenizer/tokenizer.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_language_executor_gpu.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_runtime.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

Qwen3VLDecodeExecutorGpu::Qwen3VLDecodeExecutorGpu(Qwen3VLRuntime *const runtime,
                                                   Qwen3VLGenerationContext *const context)
    : runtime_(runtime), context_(context) {
}

bool Qwen3VLDecodeExecutorGpu::Initialize() {
  OMNI_RETURN_VAL_IF(runtime_ == nullptr, SetError("Qwen3-VL decode runtime is null"));
  return true;
}

graph::Status Qwen3VLDecodeExecutorGpu::Run() {
  error_message_.clear();
  OMNI_RETURN_VAL_IF_LOG(runtime_ == nullptr || context_ == nullptr ||
                             context_->request == nullptr || context_->result == nullptr ||
                             context_->model_config == nullptr || context_->tokenizer == nullptr,
                         graph::InvalidArgumentStatus("Qwen3-VL decode context is null"), ERROR,
                         "Qwen3-VL decode context is null");
  OMNI_RETURN_VAL_IF_LOG(context_->request->max_new_tokens <= 0,
                         graph::InvalidArgumentStatus("Qwen3-VL max_new_tokens is invalid"), ERROR,
                         "Qwen3-VL max_new_tokens is invalid");
  TextStream stream(context_->request->stream_callback);
  for (int32_t token_index = 0; token_index < context_->request->max_new_tokens; ++token_index) {
    context_->result->generated_token_ids.emplace_back(context_->next_token_id);
    if (context_->next_token_id == context_->model_config->special_tokens.end_of_sequence_id ||
        context_->next_token_id == context_->model_config->special_tokens.end_of_text_id) {
      break;
    }
    std::string bytes;
    OMNI_RETURN_VAL_IF_LOG(!context_->tokenizer->Decode({context_->next_token_id}, &bytes),
                           graph::InternalStatus(context_->tokenizer->error_message()), ERROR,
                           context_->tokenizer->error_message());
    OMNI_RETURN_VAL_IF(!stream.Append(bytes), graph::InternalStatus(stream.error_message()));
    if (stream.is_cancelled() || token_index + 1 == context_->request->max_new_tokens) {
      break;
    }
    const std::vector<int32_t> token_ids = {context_->next_token_id};
    const std::vector<int32_t> position_ids = {
        runtime_->next_rope_position, runtime_->next_rope_position, runtime_->next_rope_position};
    const bool is_successful = RunQwen3VLLanguageStep(
        runtime_, token_ids, position_ids, Qwen3VLVisionFeatures{}, -1, &context_->next_token_id);
    OMNI_RETURN_VAL_IF_LOG(
        !is_successful,
        graph::InternalStatus(error_message_.empty() ? "Qwen3-VL decode failed" : error_message_),
        ERROR, error_message_);
    context_->result->decode_time_ms += runtime_->step_time_ms;
    ++context_->result->decode_token_count;
  }
  OMNI_RETURN_VAL_IF(!stream.Finish(), graph::InternalStatus(stream.error_message()));
  context_->result->text = stream.text();
  context_->result->is_cancelled = stream.is_cancelled();
  context_->result->gpu_memory_bytes = runtime_->gpu_memory_bytes;
  return graph::Status();
}

bool Qwen3VLDecodeExecutorGpu::SetError(const std::string &message) {
  error_message_ = message;
  if (context_ != nullptr) {
    context_->error_message = message;
  }
  return false;
}

const std::string &Qwen3VLDecodeExecutorGpu::error_message() const {
  return error_message_;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
