#include "Omni-Runtime/inference/engine/inference_engine.h"

#include <filesystem>
#include <utility>

#include "Omni-Runtime/utils/utils.h"
#include "Omni-Runtime/models/registry.h"

namespace omni_runtime {
namespace inference {

InferenceEngine::InferenceEngine() : has_explicit_model_(false) {
}

InferenceEngine::InferenceEngine(std::unique_ptr<Model> model)
    : has_explicit_model_(true), model_(std::move(model)) {
}

InferenceEngine::ExecutionGuard::ExecutionGuard(std::atomic_flag *const flag)
    : flag_(flag), is_acquired_(flag != nullptr && !flag->test_and_set(std::memory_order_acquire)) {
}

InferenceEngine::ExecutionGuard::~ExecutionGuard() {
  if (is_acquired_) {
    flag_->clear(std::memory_order_release);
  }
}

bool InferenceEngine::ExecutionGuard::IsAcquired() const {
  return is_acquired_;
}

bool InferenceEngine::Init(const std::string &config_path) {
  const ExecutionGuard guard(&is_busy_);
  OMNI_RETURN_VAL_IF_LOG(!guard.IsAcquired(), false, ERROR,
                         "Cannot reinitialize the pipeline from a generation callback");
  is_initialized_ = false;
  error_message_.clear();
  executor_.reset();
  if (!has_explicit_model_) {
    model_.reset();
    ModelRegistry registry;
    OMNI_RETURN_VAL_IF(!RegisterBuiltinModels(&registry) ||
                           !registry.CreateFromConfig(config_path, &model_),
                       SetError(registry.error_message()));
  }
  OMNI_RETURN_VAL_IF(model_ == nullptr, SetError("model is null"));
  OMNI_RETURN_VAL_IF(!model_->Load(config_path), SetError(model_->error_message()));
  OMNI_RETURN_VAL_IF(!model_->CreateExecutor(&executor_),
                     SetError(model_->error_message().empty() ? "model could not create an executor"
                                                              : model_->error_message()));
  OMNI_RETURN_VAL_IF(executor_ == nullptr, SetError("model returned a null executor"));
  OMNI_RETURN_VAL_IF(!executor_->Init(), SetError(executor_->error_message()));
  is_initialized_ = true;
  return true;
}

bool InferenceEngine::ValidateRequest(const ModelRequest &request) {
  if (const auto *const generation = std::get_if<GenerationRequest>(&request)) {
    OMNI_RETURN_VAL_IF(model_->descriptor().task != ModelTask::kTextGeneration,
                       SetError("model does not support text generation"));
    OMNI_RETURN_VAL_IF(generation->prompt.empty(), SetError("generation prompt is empty"));
    OMNI_RETURN_VAL_IF(generation->max_new_tokens <= 0,
                       SetError("generation max_new_tokens is invalid"));
    OMNI_RETURN_VAL_IF(!generation->image_path.empty() && !model_->descriptor().supports_images,
                       SetError("model does not support image inputs"));
    std::error_code error;
    OMNI_RETURN_VAL_IF(!generation->image_path.empty() &&
                           !std::filesystem::is_regular_file(generation->image_path, error),
                       SetError(error ? "cannot access generation image: " + error.message()
                                      : "generation image does not exist"));
    return true;
  }
  const auto &action = std::get<ActionRequest>(request);
  OMNI_RETURN_VAL_IF(model_->descriptor().task != ModelTask::kActionPrediction,
                     SetError("model does not support action prediction"));
  OMNI_RETURN_VAL_IF(action.horizon <= 0, SetError("action horizon is invalid"));
  for (const auto &observation : action.observations) {
    OMNI_RETURN_VAL_IF(observation == nullptr, SetError("action observation is null"));
  }
  return true;
}

bool InferenceEngine::Execute(const ModelRequest &request, ModelResult *const result) {
  const ExecutionGuard guard(&is_busy_);
  OMNI_RETURN_VAL_IF_LOG(!guard.IsAcquired(), false, ERROR,
                         (std::holds_alternative<GenerationRequest>(request)
                              ? "Cannot reenter generation from a stream callback"
                              : "Cannot reenter action execution"));
  OMNI_RETURN_VAL_IF(result == nullptr, SetError("model result is null"));
  OMNI_RETURN_VAL_IF(!is_initialized_ || model_ == nullptr,
                     SetError("pipeline is not initialized"));
  error_message_.clear();
  OMNI_RETURN_VAL_IF(!ValidateRequest(request), false);
  const bool is_generation = std::holds_alternative<GenerationRequest>(request);
  *result = is_generation ? ModelResult(GenerationResult{}) : ModelResult(ActionResult{});
  OMNI_RETURN_VAL_IF(!executor_->Execute(request, result), SetError(executor_->error_message()));
  OMNI_RETURN_VAL_IF(is_generation != std::holds_alternative<GenerationResult>(*result),
                     SetError("model returned an incompatible result type"));
  if (const auto *const action = std::get_if<ActionResult>(result)) {
    OMNI_RETURN_VAL_IF(
        action->horizon != std::get<ActionRequest>(request).horizon || action->action_size <= 0 ||
            action->actions.size() % static_cast<size_t>(action->action_size) != 0U ||
            action->actions.size() / static_cast<size_t>(action->action_size) !=
                static_cast<size_t>(action->horizon),
        SetError("model returned an invalid action block shape"));
  }
  return true;
}

bool InferenceEngine::Generate(const GenerationRequest &request, GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(result == nullptr, Execute(ModelRequest(request), nullptr));
  ModelResult output;
  OMNI_RETURN_VAL_IF(!Execute(ModelRequest(request), &output), false);
  *result = std::move(std::get<GenerationResult>(output));
  return true;
}

const std::string &InferenceEngine::compute_backend() const {
  static const std::string empty;
  return model_ == nullptr ? empty : model_->compute_backend();
}

const std::string &InferenceEngine::error_message() const {
  return error_message_;
}

const Model *InferenceEngine::model() const {
  return model_.get();
}

bool InferenceEngine::SetError(const std::string &message) {
  error_message_ = message;
  return false;
}

} // namespace inference
} // namespace omni_runtime
