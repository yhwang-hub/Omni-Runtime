#include "Omni-Runtime/graph/daemon/daemon_manager.h"

#include <algorithm>

#include "Omni-Runtime/utils/status_macros.h"

namespace omni_runtime {
namespace graph {

DaemonManager::~DaemonManager() {
  static_cast<void>(Destroy());
  static_cast<void>(Clear());
}

Status DaemonManager::Add(const std::shared_ptr<Daemon> &daemon) {
  OMNI_RETURN_VAL_IF_LOG(daemon == nullptr, InvalidArgumentStatus("daemon is null"), ERROR,
                    "daemon is null");
  std::lock_guard<std::mutex> lock(mutex_);
  OMNI_RETURN_VAL_IF_LOG(std::find(daemons_.begin(), daemons_.end(), daemon) != daemons_.end(),
                    InvalidArgumentStatus("daemon is duplicated"), ERROR, "daemon is duplicated");
  if (param_manager_ != nullptr && event_manager_ != nullptr) {
    OMNI_RETURN_VAL_IF_LOG(!daemon->SetContext(param_manager_, event_manager_),
                      InternalStatus("daemon context setup failed"), ERROR,
                      "daemon context setup failed");
  }
  daemons_.emplace_back(daemon);
  return Status();
}

Status DaemonManager::Remove(const std::shared_ptr<Daemon> &daemon) {
  OMNI_RETURN_VAL_IF_LOG(daemon == nullptr, InvalidArgumentStatus("daemon is null"), ERROR,
                    "daemon is null");
  OMNI_RETURN_STATUS_IF_NOT_OK(daemon->Destroy());
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = std::find(daemons_.begin(), daemons_.end(), daemon);
  OMNI_RETURN_VAL_IF_LOG(iter == daemons_.end(), NotFoundStatus("daemon not found"), ERROR,
                    "daemon not found");
  daemons_.erase(iter);
  return Status();
}

bool DaemonManager::SetContext(const std::shared_ptr<ParamManager> &param_manager,
                               const std::shared_ptr<EventManager> &event_manager) {
  OMNI_RETURN_VAL_IF(param_manager == nullptr || event_manager == nullptr, false);
  std::lock_guard<std::mutex> lock(mutex_);
  param_manager_ = param_manager;
  event_manager_ = event_manager;
  for (const auto &daemon : daemons_) {
    OMNI_RETURN_VAL_IF(!daemon->SetContext(param_manager, event_manager), false);
  }
  return true;
}

std::size_t DaemonManager::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return daemons_.size();
}

Status DaemonManager::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  daemons_.clear();
  return Status();
}

Status DaemonManager::Init() {
  std::vector<std::shared_ptr<Daemon>> daemons;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    daemons = daemons_;
  }
  for (const auto &daemon : daemons) {
    OMNI_RETURN_STATUS_IF_NOT_OK(daemon->Init());
  }
  return Status();
}

Status DaemonManager::Destroy() {
  std::vector<std::shared_ptr<Daemon>> daemons;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    daemons = daemons_;
  }
  for (auto iter = daemons.rbegin(); iter != daemons.rend(); ++iter) {
    OMNI_RETURN_STATUS_IF_NOT_OK((*iter)->Destroy());
  }
  return Status();
}

} // namespace graph
} // namespace omni_runtime
