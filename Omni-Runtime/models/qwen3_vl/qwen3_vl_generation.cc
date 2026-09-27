#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_pipeline.h"

#include <algorithm>

#include "Omni-Runtime/inference/common/text_stream/text_stream.h"
#include "Omni-Runtime/inference/tokenizer/chat_template.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

bool Qwen3VLPipeline::PreparePrompt(const GenerationRequest &request,
                                    GenerationContext *const context) {
  OMNI_RETURN_VAL_IF(context == nullptr, SetError("prompt context is null"));
  std::string prompt;
  const Qwen3VLChatTemplate chat_template;
  OMNI_RETURN_VAL_IF(
      !chat_template.BuildUserPrompt(request.prompt, !request.image_path.empty(), &prompt) ||
          !tokenizer_->Encode(prompt, &context->token_ids),
      SetError("failed to tokenize Qwen3-VL prompt: " + tokenizer_->error_message()));
  const auto &features = context->vision_features;
  if (features.embedding != nullptr) {
    const auto image = std::find(context->token_ids.begin(), context->token_ids.end(),
                                 runtime_->model_config.special_tokens.image_id);
    OMNI_RETURN_VAL_IF(image == context->token_ids.end() || features.token_count <= 0 ||
                           features.merged_grid_width <= 0 || features.merged_grid_height <= 0,
                       SetError("Qwen3-VL image placeholder or vision features are invalid"));
    context->vision_token_offset =
        static_cast<int32_t>(std::distance(context->token_ids.begin(), image));
    context->token_ids.insert(image, features.token_count - 1,
                              runtime_->model_config.special_tokens.image_id);
  }
  OMNI_RETURN_VAL_IF(
      context->token_ids.empty() ||
          context->token_ids.size() > static_cast<size_t>(runtime_->cache_capacity) ||
          request.max_new_tokens >
              runtime_->cache_capacity - static_cast<int32_t>(context->token_ids.size()),
      SetError("Qwen3-VL prompt and generation exceed context capacity"));
  OMNI_RETURN_VAL_IF(std::any_of(context->token_ids.begin(), context->token_ids.end(),
                                 [this](const int32_t token) {
                                   return token < 0 ||
                                          token >= runtime_->model_config.vocabulary_size;
                                 }),
                     SetError("Qwen3-VL token ID is outside the vocabulary"));
  const int32_t count = static_cast<int32_t>(context->token_ids.size());
  context->position_ids.assign(static_cast<size_t>(count) * 3U, 0);
  const int32_t offset = context->vision_token_offset;
  const int32_t vision_end = offset + features.token_count;
  int32_t text_position = 0;
  for (int32_t token = 0; token < count; ++token) {
    const bool is_vision = offset >= 0 && token >= offset && token < vision_end;
    if (is_vision) {
      const int32_t index = token - offset;
      const int32_t base = offset;
      context->position_ids[token] = base;
      context->position_ids[static_cast<size_t>(count) + token] =
          base + index / features.merged_grid_width;
      context->position_ids[static_cast<size_t>(count) * 2U + token] =
          base + index % features.merged_grid_width;
      if (token + 1 == vision_end) {
        text_position = base + std::max(features.merged_grid_height, features.merged_grid_width);
      }
      continue;
    }
    for (int32_t axis = 0; axis < 3; ++axis) {
      context->position_ids[static_cast<size_t>(axis) * count + token] = text_position;
    }
    ++text_position;
  }
  return true;
}

bool Qwen3VLPipeline::RunPrefill(GenerationContext *const context, GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(context == nullptr || result == nullptr, SetError("prefill outputs are null"));
  const int32_t count = static_cast<int32_t>(context->token_ids.size());
  OperatorGraphBindings bindings;
  OMNI_RETURN_VAL_IF(!BuildLanguageBindings(count, false, *context, &bindings),
                     SetError("failed to bind Qwen3-VL prefill graph"));
  const OperatorIterationBinder bind_layer =
      [this, context](const std::string &variable, const int32_t layer,
                      OperatorGraphBindings *const iteration) {
        return BindLanguageLayer(variable, layer, *context, false, iteration);
      };
  std::vector<backend::ComputeNode> embedding;
  std::vector<backend::ComputeNode> language;
  OMNI_RETURN_VAL_IF(!graph_config_.Append("embedding", bindings, bind_layer, &embedding) ||
                         !graph_config_.Append("language", bindings, bind_layer, &language),
                     SetError(graph_config_.error_message()));
  OMNI_RETURN_VAL_IF(!embedding_graph_.Prepare("qwen3_vl/embedding", runtime_->backend,
                                               RuntimeScalarType(*runtime_), embedding),
                     SetError(embedding_graph_.error_message()));
  OMNI_RETURN_VAL_IF(!language_graph_.Prepare("qwen3_vl/language", runtime_->backend,
                                              RuntimeScalarType(*runtime_), language),
                     SetError(language_graph_.error_message()));
  const auto &workspace = runtime_->language_workspaces;
  OMNI_RETURN_VAL_IF(
      !runtime_->compute_timer.Start() ||
          !workspace.tokens->CopyFromHost(context->token_ids.data(),
                                          context->token_ids.size() * sizeof(int32_t)) ||
          !workspace.positions->CopyFromHost(context->position_ids.data(),
                                             context->position_ids.size() * sizeof(int32_t)) ||
          !embedding_graph_.Run(),
      SetError("Qwen3-VL embedding graph failed: " + embedding_graph_.error_message()));
  if (context->vision_features.embedding != nullptr) {
    const size_t row_bytes =
        static_cast<size_t>(runtime_->model_config.hidden_size) * RuntimeElementSize(*runtime_);
    const size_t embedding_bytes =
        static_cast<size_t>(context->vision_features.token_count) * row_bytes;
    const memory::BufferPtr source = memory::Buffer::CreateDeviceBufferView(
        context->vision_features.embedding, 0U, embedding_bytes);
    const memory::BufferPtr destination = memory::Buffer::CreateDeviceBufferView(
        workspace.hidden, static_cast<size_t>(context->vision_token_offset) * row_bytes,
        embedding_bytes);
    OMNI_RETURN_VAL_IF(source == nullptr || destination == nullptr ||
                           !destination->CopyFromBuffer(*source),
                       SetError("failed to inject Qwen3-VL vision embedding"));
  }
  OMNI_RETURN_VAL_IF(
      !language_graph_.Run() || !runtime_->compute_timer.Stop(&runtime_->step_time_ms) ||
          !workspace.sample->CopyToHost(sizeof(context->next_token_id), &context->next_token_id),
      SetError("Qwen3-VL language graph failed: " + language_graph_.error_message()));
  runtime_->used_token_count = count;
  runtime_->next_rope_position =
      *std::max_element(context->position_ids.begin(), context->position_ids.end()) + 1;
  const int32_t positions[] = {runtime_->next_rope_position, runtime_->next_rope_position,
                               runtime_->next_rope_position};
  OMNI_RETURN_VAL_IF(!workspace.positions->CopyFromHost(positions, sizeof(positions)) ||
                         !runtime_->decode_cache_position->CopyFromHost(
                             &runtime_->used_token_count, sizeof(runtime_->used_token_count)),
                     SetError("failed to initialize Qwen3-VL resident decode positions"));
  result->backend_name =
      std::string("native_") + backend::BackendName(runtime_->backend->kind()) + "(omni-runtime)";
  result->prefill_token_count = count;
  result->vision_token_count = context->vision_features.token_count;
  result->prompt_token_ids = context->token_ids;
  result->prefill_time_ms = runtime_->step_time_ms;
  result->gpu_memory_bytes = runtime_->gpu_memory_bytes;
  return true;
}

bool Qwen3VLPipeline::RunDecode(const GenerationRequest &request, const int32_t first_token_id,
                                GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(result == nullptr, SetError("decode result is null"));
  TextStream stream(request.stream_callback);
  int32_t next_token_id = first_token_id;
  const auto &special_tokens = runtime_->model_config.special_tokens;
  for (int32_t index = 0; index < request.max_new_tokens; ++index) {
    result->generated_token_ids.push_back(next_token_id);
    if (next_token_id == special_tokens.end_of_sequence_id ||
        next_token_id == special_tokens.end_of_text_id) {
      break;
    }
    std::string bytes;
    OMNI_RETURN_VAL_IF(!tokenizer_->Decode({next_token_id}, &bytes),
                       SetError(tokenizer_->error_message()));
    OMNI_RETURN_VAL_IF(!stream.Append(bytes), SetError(stream.error_message()));
    if (stream.is_cancelled() || index + 1 == request.max_new_tokens) {
      break;
    }
    OMNI_RETURN_VAL_IF(runtime_->decode_plan == nullptr ||
                           runtime_->used_token_count >= runtime_->cache_capacity ||
                           !runtime_->decode_plan->Run(sizeof(next_token_id), &next_token_id,
                                                       &runtime_->step_time_ms),
                       SetError("Qwen3-VL decode operator graph failed"));
    ++runtime_->used_token_count;
    ++runtime_->next_rope_position;
    result->decode_time_ms += runtime_->step_time_ms;
    ++result->decode_token_count;
  }
  OMNI_RETURN_VAL_IF(!stream.Finish(), SetError(stream.error_message()));
  result->text = stream.text();
  result->is_cancelled = stream.is_cancelled();
  result->gpu_memory_bytes = runtime_->gpu_memory_bytes;
  return true;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
