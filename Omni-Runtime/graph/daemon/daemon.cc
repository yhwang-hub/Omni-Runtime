#include "Omni-Runtime/graph/daemon/daemon.h"

#include <chrono>

namespace omni_runtime {
namespace graph {

Daemon::~Daemon() {
  static_cast<void>(Destroy());
}

Milliseconds Daemon::interval_ms() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return interval_ms_;
}

bool Daemon::set_interval_ms(const Milliseconds interval_ms) {
  OMNI_RETURN_VAL_IF(interval_ms <= 0, false);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    interval_ms_ = interval_ms;
  }
  condition_.notify_all();
  return true;
}

bool Daemon::SetParam(const DaemonParam *const param) {
  if (param == nullptr) {
    param_.reset();
    return true;
  }
  param_ = param->Clone();
  return param_ != nullptr;
}

Status Daemon::Init() {
  std::lock_guard<std::mutex> lock(mutex_);
  OMNI_RETURN_VAL_IF_LOG(interval_ms_ <= 0, InvalidArgumentStatus("daemon interval is invalid"), ERROR,
                    "daemon interval is invalid: " << interval_ms_);
  OMNI_RETURN_VAL_IF_LOG(is_running_, InvalidArgumentStatus("daemon is already running"), ERROR,
                    "daemon is already running");
  is_running_ = true;
  worker_ = std::thread(&Daemon::WorkLoop, this);
  return Status();
}

Status Daemon::Destroy() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_running_ && !worker_.joinable()) {
      return Status();
    }
    is_running_ = false;
  }
  condition_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  return Status();
}

bool Daemon::SetContext(const std::shared_ptr<ParamManager> &param_manager,
                        const std::shared_ptr<EventManager> &event_manager) {
  OMNI_RETURN_VAL_IF(param_manager == nullptr || event_manager == nullptr, false);
  param_manager_ = param_manager;
  event_manager_ = event_manager;
  return true;
}

void Daemon::WorkLoop() {
  Milliseconds next_interval_ms = 0;
  while (true) {
    std::unique_lock<std::mutex> lock(mutex_);
    const Milliseconds wait_interval_ms = next_interval_ms > 0 ? next_interval_ms : interval_ms_;
    const bool is_stopped = condition_.wait_for(lock, std::chrono::milliseconds(wait_interval_ms),
                                                [this]() { return !is_running_; });
    if (is_stopped || !is_running_) {
      break;
    }
    lock.unlock();

    if (!DaemonTask(param_.get())) {
      LOG_ERROR("daemon task failed: " << name());
    }
    next_interval_ms = ModifyIntervalMs(param_.get());
  }
}

} // namespace graph
} // namespace omni_runtime
