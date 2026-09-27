#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "Omni-Runtime/models/model.h"

namespace omni_runtime {
namespace inference {

class ModelRegistry final {
public:
  using Factory = std::function<std::unique_ptr<Model>()>;

  bool Register(const std::string &name, const Factory &factory);
  bool Create(const std::string &name, std::unique_ptr<Model> *const model);
  bool CreateFromConfig(const std::string &config_path, std::unique_ptr<Model> *const model);
  const std::string &error_message() const;

private:
  bool SetError(const std::string &message);

  std::unordered_map<std::string, Factory> factories_;
  std::string error_message_;
};

bool RegisterBuiltinModels(ModelRegistry *const registry);

} // namespace inference
} // namespace omni_runtime
