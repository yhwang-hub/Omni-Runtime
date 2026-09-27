#include "Omni-Runtime/graph/stage/stage_manager.h"

#include <vector>

namespace omni_runtime {
namespace graph {

StageManager::~StageManager() {
  static_cast<void>(Clear());
}

bool StageManager::SetParamManager(const std::shared_ptr<ParamManager> &param_manager) {
  OMNI_RETURN_VAL_IF(param_manager == nullptr, false);
  std::lock_guard<std::mutex> lock(mutex_);
  param_manager_ = param_manager;
  return true;
}

Status StageManager::WaitForReady(const std::string &key) {
  std::shared_ptr<Stage> stage;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = stages_.find(key);
    OMNI_RETURN_VAL_IF_LOG(iter == stages_.end(), NotFoundStatus("stage not found: " + key), ERROR,
                      "stage not found: " << key);
    stage = iter->second;
  }
  return stage->Wait();
}

Status StageManager::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  stages_.clear();
  return Status();
}

Status StageManager::Init() {
  std::vector<std::shared_ptr<Stage>> stages;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &item : stages_) {
      stages.emplace_back(item.second);
    }
  }
  for (const auto &stage : stages) {
    OMNI_RETURN_STATUS_IF_NOT_OK(stage->Init());
  }
  return Status();
}

Status StageManager::Destroy() {
  std::vector<std::shared_ptr<Stage>> stages;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &item : stages_) {
      stages.emplace_back(item.second);
    }
  }
  for (auto iter = stages.rbegin(); iter != stages.rend(); ++iter) {
    OMNI_RETURN_STATUS_IF_NOT_OK((*iter)->Destroy());
  }
  return Status();
}

} // namespace graph
} // namespace omni_runtime
