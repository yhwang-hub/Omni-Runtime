#include "Omni-Runtime/graph/param/param_manager.h"

#include "Omni-Runtime/utils/status_macros.h"

namespace omni_runtime {
namespace graph {

ParamManager::~ParamManager() {
  static_cast<void>(Clear());
}

Status ParamManager::Remove(const std::string &key) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = params_.find(key);
  OMNI_RETURN_VAL_IF_LOG(iter == params_.end(), NotFoundStatus("param not found: " + key), ERROR,
                    "param not found: " << key);
  params_.erase(iter);
  return Status();
}

Status ParamManager::GetKeys(std::vector<std::string> *const keys) const {
  OMNI_RETURN_VAL_IF_LOG(keys == nullptr, InvalidArgumentStatus("keys is null"), ERROR, "keys is null");
  std::lock_guard<std::mutex> lock(mutex_);
  keys->clear();
  keys->reserve(params_.size());
  for (const auto &item : params_) {
    keys->emplace_back(item.first);
  }
  return Status();
}

bool ParamManager::Has(const std::string &key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return params_.find(key) != params_.end();
}

Status ParamManager::Setup() {
  std::vector<std::shared_ptr<Param>> params;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &item : params_) {
      params.emplace_back(item.second);
    }
  }
  for (const auto &param : params) {
    OMNI_RETURN_STATUS_IF_NOT_OK(param->Setup());
  }
  return Status();
}

bool ParamManager::Reset(const Status &status) {
  std::vector<std::shared_ptr<Param>> params;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &item : params_) {
      params.emplace_back(item.second);
    }
  }
  for (const auto &param : params) {
    OMNI_RETURN_VAL_IF(!param->Reset(status), false);
  }
  return true;
}

Status ParamManager::Init() {
  std::vector<std::shared_ptr<Param>> params;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &item : params_) {
      params.emplace_back(item.second);
    }
  }
  for (const auto &param : params) {
    OMNI_RETURN_STATUS_IF_NOT_OK(param->Init());
  }
  return Status();
}

Status ParamManager::Destroy() {
  std::vector<std::shared_ptr<Param>> params;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &item : params_) {
      params.emplace_back(item.second);
    }
  }
  for (auto iter = params.rbegin(); iter != params.rend(); ++iter) {
    OMNI_RETURN_STATUS_IF_NOT_OK((*iter)->Destroy());
  }
  return Status();
}

Status ParamManager::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  params_.clear();
  return Status();
}

} // namespace graph
} // namespace omni_runtime
