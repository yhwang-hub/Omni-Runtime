#include "Omni-Runtime/models/qwen3_vl/executor.h"
#include "Omni-Runtime/models/qwen3_vl/model.h"

#include <algorithm>

#include "Omni-Runtime/inference/common/image_preprocessor/image_preprocessor.h"
#include "Omni-Runtime/inference/generation/autoregressive_runner.h"
#include "Omni-Runtime/models/components/qwen_vision/graph_binding.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

Qwen3VLExecutor::Qwen3VLExecutor(const Qwen3VLModel &model)
    : model_(model), graph_config_(model.graph_config()) {
}

bool Qwen3VLExecutor::Init() {
  ClearError();
  context_ = std::make_unique<ExecutionContext>();
  state_ = RequestState{};
  context_->gpu_memory_bytes = model_.data().gpu_memory_bytes;
  OMNI_RETURN_VAL_IF(!model_.data().has_weights_loaded ||
                         !context_->compute_timer.Initialize(model_.data().backend) ||
                         !AllocateKvCache(model_.data().config.max_context_tokens()) ||
                         !AllocateWorkspaces(),
                     SetError("failed to initialize Qwen3-VL compute storage"));
  context_->gpu_memory_bytes += model_.data().backend->compute_workspace_bytes();
  OMNI_RETURN_VAL_IF(!BuildGraph(), SetError(error_message().empty() ? graph_config_.error_message()
                                                                     : error_message()));
  return true;
}

bool Qwen3VLExecutor::Execute(const ModelRequest &request, ModelResult *const result) {
  OMNI_RETURN_VAL_IF(result == nullptr, SetError("generation result is null"));
  const auto *const generation = std::get_if<GenerationRequest>(&request);
  auto *const output = std::get_if<GenerationResult>(result);
  OMNI_RETURN_VAL_IF(generation == nullptr || output == nullptr,
                     SetError("Qwen3-VL requires a text generation request and result"));
  ClearError();
  OMNI_RETURN_VAL_IF(context_ == nullptr || context_->decode_plan == nullptr,
                     SetError("Qwen3-VL executor is not initialized"));
  OMNI_RETURN_VAL_IF(!ResetState(), SetError("failed to reset generation"));
  GenerationContext context;
  OMNI_RETURN_VAL_IF(
      !RunVision(*generation, &context, output) || !PreparePrompt(*generation, &context) ||
          !RunPrefill(&context, output) || !RunDecode(*generation, context.next_token_id, output),
      SetError(error_message().empty() ? graph_config_.error_message() : error_message()));
  return true;
}

bool Qwen3VLExecutor::PreparePrompt(const GenerationRequest &request,
                                    GenerationContext *const context) {
  OMNI_RETURN_VAL_IF(context == nullptr, SetError("prompt context is null"));
  const auto &features = context->vision_features;
  if (features.embedding != nullptr) {
    OMNI_RETURN_VAL_IF(features.token_count <= 0,
                       SetError("Qwen3-VL image placeholder or vision features are invalid"));
    context->image_tokens = features.token_count;
    context->image_grid_height = features.merged_grid_height;
    context->image_grid_width = features.merged_grid_width;
  }
  std::string processor_error;
  OMNI_RETURN_VAL_IF(!model_.processor().Prepare(
                         request, model_.data().model_config.special_tokens.image_id,
                         model_.data().model_config.vocabulary_size,
                         model_.data().config.max_context_tokens(), context, &processor_error),
                     SetError(processor_error));
  return true;
}

bool Qwen3VLExecutor::RunPrefill(GenerationContext *const context, GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(context == nullptr || result == nullptr, SetError("prefill outputs are null"));
  const int32_t count = static_cast<int32_t>(context->tokens.size());
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
  OMNI_RETURN_VAL_IF(
      !graph_config_.AppendSections({"embedding"}, bindings, bind_layer, &embedding) ||
          !graph_config_.AppendSections({"language"}, bindings, bind_layer, &language),
      false);
  OMNI_RETURN_VAL_IF(!context_->embedding_graph.Prepare("qwen3_vl/embedding", model_.data().backend,
                                                        RuntimeScalarType(), embedding),
                     SetError(context_->embedding_graph.error_message()));
  OMNI_RETURN_VAL_IF(!context_->language_graph.Prepare("qwen3_vl/language", model_.data().backend,
                                                       RuntimeScalarType(), language),
                     SetError(context_->language_graph.error_message()));
  const auto &workspace = context_->language_workspaces;
  OMNI_RETURN_VAL_IF(
      !context_->compute_timer.Start() ||
          !workspace.tokens->CopyFromHost(context->tokens.data(),
                                          context->tokens.size() * sizeof(int32_t)) ||
          !workspace.positions->CopyFromHost(context->positions.data(),
                                             context->positions.size() * sizeof(int32_t)) ||
          !context_->embedding_graph.Run(),
      SetError("Qwen3-VL embedding graph failed: " + context_->embedding_graph.error_message()));
  if (context->vision_features.embedding != nullptr) {
    const size_t row_bytes =
        static_cast<size_t>(model_.data().model_config.hidden_size) * RuntimeElementSize();
    const size_t embedding_bytes =
        static_cast<size_t>(context->vision_features.token_count) * row_bytes;
    const memory::BufferPtr source = memory::Buffer::CreateDeviceBufferView(
        context->vision_features.embedding, 0U, embedding_bytes);
    const memory::BufferPtr destination = memory::Buffer::CreateDeviceBufferView(
        workspace.hidden, static_cast<size_t>(context->image_offset) * row_bytes, embedding_bytes);
    OMNI_RETURN_VAL_IF(source == nullptr || destination == nullptr ||
                           !destination->CopyFromBuffer(*source),
                       SetError("failed to inject Qwen3-VL vision embedding"));
  }
  OMNI_RETURN_VAL_IF(
      !context_->language_graph.Run() || !context_->compute_timer.Stop(&context_->step_time_ms) ||
          !workspace.sample->CopyToHost(sizeof(context->next_token_id), &context->next_token_id),
      SetError("Qwen3-VL language graph failed: " + context_->language_graph.error_message()));
  state_.used_token_count = count;
  state_.next_rope_position =
      *std::max_element(context->positions.begin(), context->positions.end()) + 1;
  const int32_t positions[] = {state_.next_rope_position, state_.next_rope_position,
                               state_.next_rope_position};
  OMNI_RETURN_VAL_IF(!workspace.positions->CopyFromHost(positions, sizeof(positions)) ||
                         !state_.decode_cache_position->CopyFromHost(
                             &state_.used_token_count, sizeof(state_.used_token_count)),
                     SetError("failed to initialize Qwen3-VL resident decode positions"));
  result->backend_name = std::string("native_") +
                         backend::BackendName(model_.data().backend->kind()) + "(omni-runtime)";
  result->prefill_token_count = count;
  result->vision_token_count = context->vision_features.token_count;
  result->prompt_token_ids = context->tokens;
  result->prefill_time_ms = context_->step_time_ms;
  result->gpu_memory_bytes = context_->gpu_memory_bytes;
  return true;
}

bool Qwen3VLExecutor::RunDecode(const GenerationRequest &request, const int32_t first_token_id,
                                GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(result == nullptr, SetError("decode result is null"));
  const auto &special_tokens = model_.data().model_config.special_tokens;
  const AutoregressiveRunner::DecodeStep decode_step = [this](int32_t *const token,
                                                              double *const elapsed_ms) {
    return RunDecodeStep(token, elapsed_ms);
  };
  std::string runner_error;
  OMNI_RETURN_VAL_IF(!AutoregressiveRunner::Run(
                         request, first_token_id, model_.data().model_config.vocabulary_size,
                         special_tokens.end_of_sequence_id, special_tokens.end_of_text_id,
                         *model_.processor().tokenizer(), decode_step, result, &runner_error),
                     SetError(error_message().empty() ? runner_error : error_message()));
  result->gpu_memory_bytes = context_->gpu_memory_bytes;
  return true;
}

bool Qwen3VLExecutor::RunDecodeStep(int32_t *const token, double *const elapsed_ms) {
  OMNI_RETURN_VAL_IF(token == nullptr || elapsed_ms == nullptr || context_ == nullptr ||
                         context_->decode_plan == nullptr ||
                         state_.used_token_count >= state_.cache_capacity ||
                         !context_->decode_plan->Run(sizeof(*token), token, elapsed_ms),
                     SetError("Qwen3-VL decode operator graph failed"));
  context_->step_time_ms = *elapsed_ms;
  ++state_.used_token_count;
  ++state_.next_rope_position;
  return true;
}

bool Qwen3VLExecutor::BuildGraph() {
  const GenerationContext context;
  OperatorGraphBindings bindings;
  OMNI_RETURN_VAL_IF(!BuildLanguageBindings(1, true, context, &bindings),
                     SetError("failed to bind Qwen3-VL decode graph"));
  const OperatorIterationBinder bind_layer =
      [this, &context](const std::string &variable, const int32_t layer,
                       OperatorGraphBindings *const iteration) {
        return BindLanguageLayer(variable, layer, context, true, iteration);
      };
  std::vector<backend::ComputeNode> nodes;
  OMNI_RETURN_VAL_IF(!graph_config_.AppendSections({"embedding", "language", "decode"}, bindings,
                                                   bind_layer, &nodes),
                     false);
  OMNI_RETURN_VAL_IF(!model_.data().backend->CreateExecutionPlan(
                         RuntimeScalarType(), nodes,
                         context_->language_workspaces.sample->allocation(),
                         &context_->decode_plan),
                     SetError("failed to prepare Qwen3-VL decode operator graph"));
  return true;
}

bool Qwen3VLExecutor::RunVision(const GenerationRequest &request, GenerationContext *const context,
                                GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(context == nullptr || result == nullptr, SetError("vision outputs are null"));
  if (request.image_path.empty()) {
    return true;
  }
  std::vector<uint8_t> pixels;
  int32_t width_pixels = 0;
  int32_t height_pixels = 0;
  ImagePatchGrid grid;
  const auto &config = model_.data().vision_config;
  OMNI_RETURN_VAL_IF(
      !ImagePreprocessor::LoadImage(request.image_path, &pixels, &width_pixels, &height_pixels) ||
          !ImagePreprocessor::MakeGrid(width_pixels, height_pixels, config.patch_size,
                                       config.merge_size, &grid),
      SetError("failed to prepare Qwen3-VL image"));
  OMNI_RETURN_VAL_IF(grid.merged_token_count > state_.cache_capacity,
                     SetError("Qwen3-VL image exceeds workspace capacity"));
  const auto &workspace = context_->vision_workspaces;
  OperatorGraphBindings bindings =
      MakeOperatorGraphBindings([this](const std::string &name) { return View(name); });
  bindings.values = {{"source", workspace.source->view()},
                     {"patches", workspace.patches->view()},
                     {"patch_embedding", workspace.patch_embedding->view()},
                     {"hidden", workspace.hidden->view()},
                     {"residual", workspace.residual->view()},
                     {"normalized", workspace.normalized->view()},
                     {"qkv", workspace.qkv->view()},
                     {"query", workspace.query->view()},
                     {"key", workspace.key->view()},
                     {"value", workspace.value->view()},
                     {"attention", workspace.attention->view()},
                     {"projected", workspace.projected->view()},
                     {"feed_forward", workspace.feed_forward->view()},
                     {"embedding", workspace.embedding->view()},
                     {"projector_first_weight", View("mm.0.weight")},
                     {"projector_first_bias", View("mm.0.bias")},
                     {"projector_second_weight", View("mm.2.weight")},
                     {"projector_second_bias", View("mm.2.bias")}};
  OMNI_RETURN_VAL_IF(
      !qwen3::BindVisionDimensions(config, grid, width_pixels, height_pixels, &bindings), false);
  const OperatorIterationBinder bind_layer =
      [this, &config, &workspace](const std::string &variable, const int32_t layer,
                                  OperatorGraphBindings *const iteration) {
        OMNI_RETURN_VAL_IF(iteration == nullptr, false);
        if (variable != "vision_layer") {
          return true;
        }
        const auto found =
            std::find(config.deepstack_layers.begin(), config.deepstack_layers.end(), layer);
        const bool has_deepstack = found != config.deepstack_layers.end();
        iteration->values["has_deepstack"] = has_deepstack;
        if (has_deepstack) {
          const size_t index =
              static_cast<size_t>(std::distance(config.deepstack_layers.begin(), found));
          iteration->values["embedding"] = workspace.deepstack[index]->view();
          const std::string prefix = "v.deepstack." + std::to_string(layer) + ".";
          iteration->values["projector_first_weight"] = View(prefix + "fc1.weight");
          iteration->values["projector_first_bias"] = View(prefix + "fc1.bias");
          iteration->values["projector_second_weight"] = View(prefix + "fc2.weight");
          iteration->values["projector_second_bias"] = View(prefix + "fc2.bias");
        }
        return true;
      };
  std::vector<backend::ComputeNode> nodes;
  OMNI_RETURN_VAL_IF(!graph_config_.AppendSections({"vision"}, bindings, bind_layer, &nodes),
                     false);
  OMNI_RETURN_VAL_IF(!context_->vision_graph.Prepare("qwen3_vl/vision", model_.data().backend,
                                                     RuntimeScalarType(), nodes),
                     SetError(context_->vision_graph.error_message()));
  OMNI_RETURN_VAL_IF(
      !context_->compute_timer.Start() ||
          !workspace.source->CopyFromHost(pixels.data(), pixels.size()) ||
          !context_->vision_graph.Run() || !context_->compute_timer.Stop(&context_->step_time_ms),
      SetError("Qwen3-VL vision operator graph failed: " + context_->vision_graph.error_message()));
  auto &features = context->vision_features;
  features.embedding = workspace.embedding;
  features.deepstack_embeddings = workspace.deepstack;
  features.token_count = grid.merged_token_count;
  features.merged_grid_height = grid.height_patches / config.merge_size;
  features.merged_grid_width = grid.width_patches / config.merge_size;
  features.input_image_size = {.width_pixels = width_pixels, .height_pixels = height_pixels};
  features.resized_image_size = {.width_pixels = grid.width_patches * config.patch_size,
                                 .height_pixels = grid.height_patches * config.patch_size};
  result->vision_time_ms = context_->step_time_ms;
  result->input_image_size = features.input_image_size;
  result->resized_image_size = features.resized_image_size;
  return true;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
