#include "Omni-Runtime/models/qwen3_5/executor.h"
#include "Omni-Runtime/models/qwen3_5/model.h"

#include <algorithm>
#include <utility>

#include "Omni-Runtime/utils/utils.h"
#include "Omni-Runtime/inference/common/image_preprocessor/image_preprocessor.h"
#include "Omni-Runtime/inference/generation/autoregressive_runner.h"
#include "Omni-Runtime/models/components/qwen_vision/graph_binding.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_5 {

backend::ScalarType Qwen35Executor::ComputeType() const {
  return model_.config().compute_type() == ModelWeightType::kBFloat16
             ? backend::ScalarType::kBFloat16
             : backend::ScalarType::kFloat32;
}

Qwen35Executor::Qwen35Executor(const Qwen35Model &model)
    : model_(model), graph_config_(model.graph_config()) {
}

bool Qwen35Executor::Init() {
  ClearError();
  context_.decode_plan.reset();
  context_.timer.reset();
  context_.buffers.clear();
  state_.layers.clear();
  context_.backend = model_.backend();
  context_.gpu_memory_bytes = model_.gpu_memory_bytes();
  context_.timer = std::make_unique<backend::EventTimer>();
  OMNI_RETURN_VAL_IF(!context_.timer->Initialize(context_.backend) || !AllocateStorage() ||
                         !ResetState() || !PrepareDecode(),
                     SetError(error_message().empty() ? "cannot initialize model execution storage"
                                                      : error_message()));
  return true;
}

bool Qwen35Executor::Execute(const ModelRequest &request, ModelResult *const result) {
  OMNI_RETURN_VAL_IF(result == nullptr, SetError("generation output is null"));
  const auto *const generation = std::get_if<GenerationRequest>(&request);
  auto *const output = std::get_if<GenerationResult>(result);
  OMNI_RETURN_VAL_IF(generation == nullptr || output == nullptr,
                     SetError("Qwen3.5 requires a text generation request and result"));
  ClearError();
  OMNI_RETURN_VAL_IF(context_.decode_plan == nullptr || model_.processor().tokenizer() == nullptr,
                     SetError("Qwen3.5 model is not initialized"));
  OMNI_RETURN_VAL_IF(generation->max_new_tokens >= model_.config().max_context_tokens(),
                     SetError("invalid prompt or maximum generated token count"));
  OMNI_RETURN_VAL_IF(!ResetState(), false);
  qwen3::Prompt prompt;
  int32_t first_token = -1;
  OMNI_RETURN_VAL_IF(
      !EncodeImage(*generation, &prompt, output),
      SetError(error_message().empty() ? graph_config_.error_message() : error_message()));
  std::string processor_error;
  OMNI_RETURN_VAL_IF(!model_.processor().Prepare(*generation, model_.model_config().image_id,
                                                 model_.model_config().vocabulary_size,
                                                 model_.config().max_context_tokens(), &prompt,
                                                 &processor_error),
                     SetError(processor_error));
  OMNI_RETURN_VAL_IF(
      !Prefill(prompt, &first_token, output),
      SetError(error_message().empty() ? graph_config_.error_message() : error_message()));
  const AutoregressiveRunner::DecodeStep decode_step = [this](int32_t *const token,
                                                              double *const elapsed_ms) {
    return RunDecodeStep(token, elapsed_ms);
  };
  std::string runner_error;
  OMNI_RETURN_VAL_IF(!AutoregressiveRunner::Run(
                         *generation, first_token, model_.model_config().vocabulary_size,
                         model_.model_config().end_of_sequence_id,
                         model_.model_config().end_of_text_id, *model_.processor().tokenizer(),
                         decode_step, output, &runner_error),
                     SetError(error_message().empty() ? runner_error : error_message()));
  return true;
}

bool Qwen35Executor::RunDecodeStep(int32_t *const token, double *const elapsed_ms) {
  OMNI_RETURN_VAL_IF(token == nullptr || elapsed_ms == nullptr || context_.decode_plan == nullptr ||
                         !context_.decode_plan->Run(sizeof(*token), token, elapsed_ms),
                     SetError("resident decode graph execution failed"));
  return true;
}

bool Qwen35Executor::PrepareDecode() {
  std::vector<backend::ComputeNode> nodes;
  OMNI_RETURN_VAL_IF(!MakeLanguageNodes(1, 0, true, nullptr, &nodes), false);
  OMNI_RETURN_VAL_IF(
      !context_.backend->CreateExecutionPlan(
          ComputeType(), nodes, context_.buffers.at("sample")->allocation(), &context_.decode_plan),
      SetError("failed to prepare resident decode graph"));
  return true;
}

bool Qwen35Executor::Prefill(const qwen3::Prompt &prompt, int32_t *const next_token,
                             GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(next_token == nullptr || result == nullptr,
                     SetError("prefill output is null"));
  const int32_t total = static_cast<int32_t>(prompt.tokens.size());
  const auto &c = model_.model_config();
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
    OMNI_RETURN_VAL_IF(
        !context_.backend->CreateExecutionPlan(ComputeType(), nodes,
                                               context_.buffers.at("sample")->allocation(), &plan),
        SetError("failed to prepare prefill graph"));
    std::vector<int32_t> positions(static_cast<size_t>(count) * 3U);
    for (int32_t axis = 0; axis < 3; ++axis) {
      std::copy_n(prompt.positions.begin() + static_cast<size_t>(axis) * total + first, count,
                  positions.begin() + static_cast<size_t>(axis) * count);
    }
    OMNI_RETURN_VAL_IF(!context_.timer->Start() ||
                           !context_.buffers.at("tokens")->CopyFromHost(
                               prompt.tokens.data() + first, count * sizeof(int32_t)) ||
                           !context_.buffers.at("positions")->CopyDataFromVector(positions),
                       SetError("failed to execute token embedding"));
    for (const auto &node : embedding) {
      OMNI_RETURN_VAL_IF(!context_.backend->Compute(ComputeType(), node.operation),
                         SetError("failed to execute token embedding"));
    }
    if (prompt.image_offset >= 0 && first >= prompt.image_offset &&
        first < prompt.image_offset + prompt.image_tokens) {
      const size_t scalar_bytes = backend::ElementBytes(ComputeType());
      const size_t bytes = static_cast<size_t>(count) * c.hidden_size * scalar_bytes;
      const auto source = memory::Buffer::CreateDeviceBufferView(
          context_.buffers.at("image_embedding"),
          static_cast<size_t>(first - prompt.image_offset) * c.hidden_size * scalar_bytes, bytes);
      const auto destination =
          memory::Buffer::CreateDeviceBufferView(context_.buffers.at("hidden"), 0U, bytes);
      OMNI_RETURN_VAL_IF(source == nullptr || destination == nullptr ||
                             !destination->CopyFromBuffer(*source),
                         SetError("failed to inject vision embeddings"));
    }
    double graph_ms = 0.0;
    double elapsed_ms = 0.0;
    OMNI_RETURN_VAL_IF(!plan->Run(sizeof(*next_token), next_token, &graph_ms) ||
                           !context_.timer->Stop(&elapsed_ms),
                       SetError("prefill execution failed"));
    result->prefill_time_ms += elapsed_ms;
    first += count;
  }
  const int32_t positions[] = {prompt.next_position, prompt.next_position, prompt.next_position};
  OMNI_RETURN_VAL_IF(
      !context_.buffers.at("positions")->CopyFromHost(positions, sizeof(positions)) ||
          !context_.buffers.at("cache_position")->CopyFromHost(&total, sizeof(total)),
      SetError("failed to initialize decode positions"));
  result->backend_name = "native_" + model_.compute_backend() + "(omni-runtime)";
  result->prefill_token_count = total;
  result->prompt_token_ids = prompt.tokens;
  result->vision_token_count = prompt.image_tokens;
  result->gpu_memory_bytes =
      context_.gpu_memory_bytes + context_.backend->compute_workspace_bytes();
  return true;
}

bool Qwen35Executor::EncodeImage(const GenerationRequest &request, qwen3::Prompt *const prompt,
                                 GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(prompt == nullptr || result == nullptr, SetError("vision output is null"));
  if (request.image_path.empty()) {
    return true;
  }
  const auto &v = model_.model_config().vision;
  std::vector<uint8_t> pixels;
  int32_t width_pixels = 0;
  int32_t height_pixels = 0;
  ImagePatchGrid grid;
  OMNI_RETURN_VAL_IF(!ImagePreprocessor::LoadImage(request.image_path,
                                                   !model_.model_config().has_fp8_weights, &pixels,
                                                   &width_pixels, &height_pixels) ||
                         !ImagePreprocessor::MakeGrid(width_pixels, height_pixels, v.patch_size,
                                                      v.merge_size, &grid),
                     SetError("cannot prepare input image"));
  OMNI_RETURN_VAL_IF(grid.merged_token_count > model_.config().max_context_tokens() ||
                         pixels.size() > context_.buffers.at("vision_source")->size_bytes(),
                     SetError("input image exceeds allocated context storage"));
  auto bindings = Bindings();
  OMNI_RETURN_VAL_IF(!qwen3::BindVisionDimensions(v, grid, width_pixels, height_pixels, &bindings),
                     false);
  std::vector<backend::ComputeNode> nodes;
  OMNI_RETURN_VAL_IF(!graph_config_.AppendSections({"vision"}, bindings, {}, &nodes), false);
  const auto readback = memory::Buffer::CreateDeviceBufferView(
      context_.buffers.at("image_embedding"), 0U, sizeof(float));
  std::unique_ptr<backend::ExecutionPlan> plan;
  OMNI_RETURN_VAL_IF(readback == nullptr ||
                         !context_.backend->CreateExecutionPlan(ComputeType(), nodes,
                                                                readback->allocation(), &plan),
                     SetError("failed to prepare vision execution graph"));
  float first_embedding = 0.0F;
  double graph_time_ms = 0.0;
  OMNI_RETURN_VAL_IF(!context_.timer->Start() ||
                         !context_.buffers.at("vision_source")->CopyDataFromVector(pixels) ||
                         !plan->Run(sizeof(first_embedding), &first_embedding, &graph_time_ms) ||
                         !context_.timer->Stop(&result->vision_time_ms),
                     SetError("vision execution failed"));
  prompt->image_tokens = grid.merged_token_count;
  prompt->image_grid_height = grid.height_patches / v.merge_size;
  prompt->image_grid_width = grid.width_patches / v.merge_size;
  result->input_image_size = {width_pixels, height_pixels};
  result->resized_image_size = {grid.width_patches * v.patch_size,
                                grid.height_patches * v.patch_size};
  return true;
}

} // namespace qwen3_5
} // namespace inference
} // namespace omni_runtime
