#include "Omni-Runtime/graph/aspect/aspect_manager.h"

#include "Omni-Runtime/utils/status_macros.h"

namespace omni_runtime {
namespace graph {

Status AspectManager::Add(const std::shared_ptr<Aspect> &aspect, const AspectParam *const param) {
  OMNI_RETURN_VAL_IF_LOG(aspect == nullptr, InvalidArgumentStatus("aspect is null"), ERROR,
                    "aspect is null");
  OMNI_RETURN_VAL_IF_LOG(!aspect->SetParam(param), InternalStatus("aspect param clone failed"), ERROR,
                    "aspect param clone failed");
  std::lock_guard<std::mutex> lock(mutex_);
  if (param_manager_ != nullptr && event_manager_ != nullptr) {
    OMNI_RETURN_VAL_IF_LOG(!aspect->SetContext(param_manager_, event_manager_, belong_name_),
                      InternalStatus("aspect context setup failed"), ERROR,
                      "aspect context setup failed");
  }
  aspects_.emplace_back(aspect);
  return Status();
}

Status AspectManager::Reflect(const AspectType type, const Status &current_status) const {
  std::vector<std::shared_ptr<Aspect>> aspects;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    aspects = aspects_;
  }
  for (const auto &aspect : aspects) {
    switch (type) {
    case AspectType::kBeginInit:
      OMNI_RETURN_STATUS_IF_NOT_OK(aspect->BeginInit());
      break;
    case AspectType::kFinishInit:
      OMNI_RETURN_VAL_IF_LOG(!aspect->FinishInit(current_status),
                        InternalStatus("finish init aspect failed"), ERROR,
                        "finish init aspect failed");
      break;
    case AspectType::kBeginRun:
      OMNI_RETURN_STATUS_IF_NOT_OK(aspect->BeginRun());
      break;
    case AspectType::kFinishRun:
      OMNI_RETURN_VAL_IF_LOG(!aspect->FinishRun(current_status),
                        InternalStatus("finish run aspect failed"), ERROR,
                        "finish run aspect failed");
      break;
    case AspectType::kBeginDestroy:
      OMNI_RETURN_STATUS_IF_NOT_OK(aspect->BeginDestroy());
      break;
    case AspectType::kFinishDestroy:
      OMNI_RETURN_VAL_IF_LOG(!aspect->FinishDestroy(current_status),
                        InternalStatus("finish destroy aspect failed"), ERROR,
                        "finish destroy aspect failed");
      break;
    case AspectType::kEnterTimeout:
      OMNI_RETURN_VAL_IF_LOG(!aspect->EnterTimeout(), InternalStatus("timeout aspect failed"), ERROR,
                        "timeout aspect failed");
      break;
    case AspectType::kEnterCrashed:
      OMNI_RETURN_VAL_IF_LOG(!aspect->EnterCrashed(), InternalStatus("crashed aspect failed"), ERROR,
                        "crashed aspect failed");
      break;
    }
  }
  return Status();
}

Status AspectManager::PopLast() {
  std::lock_guard<std::mutex> lock(mutex_);
  OMNI_RETURN_VAL_IF_LOG(aspects_.empty(), NotFoundStatus("aspect list is empty"), ERROR,
                    "aspect list is empty");
  aspects_.pop_back();
  return Status();
}

bool AspectManager::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  aspects_.clear();
  return true;
}

std::size_t AspectManager::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return aspects_.size();
}

bool AspectManager::SetContext(const std::shared_ptr<ParamManager> &param_manager,
                               const std::shared_ptr<EventManager> &event_manager,
                               const std::string &belong_name) {
  OMNI_RETURN_VAL_IF(param_manager == nullptr || event_manager == nullptr, false);
  std::lock_guard<std::mutex> lock(mutex_);
  param_manager_ = param_manager;
  event_manager_ = event_manager;
  belong_name_ = belong_name;
  for (const auto &aspect : aspects_) {
    OMNI_RETURN_VAL_IF(!aspect->SetContext(param_manager, event_manager, belong_name), false);
  }
  return true;
}

} // namespace graph
} // namespace omni_runtime
