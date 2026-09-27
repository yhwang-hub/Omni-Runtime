#pragma once

#include <string>
#include <memory>

#include "Omni-Runtime/models/model_define.h"

namespace omni_runtime {
namespace inference {

class ModelExecutor;

class Model {
public:
  virtual ~Model() = default;
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  virtual bool Load(const std::string &config_path) = 0;
  virtual bool CreateExecutor(std::unique_ptr<ModelExecutor> *const executor) const = 0;
  virtual const ModelDescriptor &descriptor() const = 0;
  virtual const std::string &compute_backend() const = 0;

  const std::string &error_message() const {
    return error_message_;
  }

protected:
  Model() = default;

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
