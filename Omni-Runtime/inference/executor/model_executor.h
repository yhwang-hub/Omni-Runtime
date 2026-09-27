#pragma once

#include <string>

#include "Omni-Runtime/inference/executor/executor_define.h"

namespace omni_runtime {
namespace inference {

class ModelExecutor {
public:
  virtual ~ModelExecutor() = default;
  ModelExecutor(const ModelExecutor &) = delete;
  ModelExecutor &operator=(const ModelExecutor &) = delete;

  virtual bool Init() = 0;
  virtual bool Execute(const ModelRequest &request, ModelResult *const result) = 0;

  const std::string &error_message() const {
    return error_message_;
  }

protected:
  ModelExecutor() = default;

  bool SetError(const std::string &message) {
    error_message_ = message;
    return false;
  }

  bool ClearError() {
    error_message_.clear();
    return true;
  }

private:
  std::string error_message_;
};

} // namespace inference
} // namespace omni_runtime
