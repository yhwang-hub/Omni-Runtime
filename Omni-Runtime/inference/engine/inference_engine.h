#pragma once

#include <atomic>
#include <string>

#include <memory>

#include "Omni-Runtime/models/model.h"
#include "Omni-Runtime/inference/executor/model_executor.h"

namespace omni_runtime {
namespace inference {

class InferenceEngine final {
public:
  InferenceEngine();
  explicit InferenceEngine(std::unique_ptr<Model> model);
  InferenceEngine(const InferenceEngine &) = delete;
  InferenceEngine &operator=(const InferenceEngine &) = delete;

  bool Init(const std::string &config_path);
  bool Execute(const ModelRequest &request, ModelResult *const result);
  bool Generate(const GenerationRequest &request, GenerationResult *const result);
  const std::string &compute_backend() const;
  const std::string &error_message() const;
  const Model *model() const;

private:
  class ExecutionGuard final {
  public:
    explicit ExecutionGuard(std::atomic_flag *const flag);
    ~ExecutionGuard();
    bool IsAcquired() const;

  private:
    std::atomic_flag *const flag_;
    const bool is_acquired_;
  };

  bool ValidateRequest(const ModelRequest &request);
  bool SetError(const std::string &message);

  const bool has_explicit_model_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<ModelExecutor> executor_;
  std::atomic_flag is_busy_ = ATOMIC_FLAG_INIT;
  bool is_initialized_ = false;
  std::string error_message_;
};

} // namespace inference
} // namespace omni_runtime
