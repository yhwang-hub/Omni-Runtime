#include "Omni-Runtime/models/registry.h"

#include "Omni-Runtime/inference/common/inference_config/inference_config.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {

bool ModelRegistry::Register(const std::string &name, const Factory &factory) {
  error_message_.clear();
  OMNI_RETURN_VAL_IF(name.empty() || !factory, SetError("model name or factory is empty"));
  OMNI_RETURN_VAL_IF(factories_.contains(name), SetError("duplicate model registration: " + name));
  factories_.emplace(name, factory);
  return true;
}

bool ModelRegistry::Create(const std::string &name, std::unique_ptr<Model> *const model) {
  OMNI_RETURN_VAL_IF(model == nullptr, SetError("model output is null"));
  error_message_.clear();
  const auto found = factories_.find(name);
  OMNI_RETURN_VAL_IF(found == factories_.end(), SetError("unsupported model: " + name));
  auto instance = found->second();
  OMNI_RETURN_VAL_IF(instance == nullptr, SetError("model factory returned null: " + name));
  *model = std::move(instance);
  return true;
}

bool ModelRegistry::CreateFromConfig(const std::string &config_path,
                                     std::unique_ptr<Model> *const model) {
  OMNI_RETURN_VAL_IF(model == nullptr, SetError("model output is null"));
  InferenceConfig config;
  OMNI_RETURN_VAL_IF(!config.Load(config_path), SetError(config.error_message()));
  return Create(config.model_name(), model);
}

const std::string &ModelRegistry::error_message() const {
  return error_message_;
}

bool ModelRegistry::SetError(const std::string &message) {
  error_message_ = message;
  LOG_ERROR(message);
  return false;
}

} // namespace inference
} // namespace omni_runtime
