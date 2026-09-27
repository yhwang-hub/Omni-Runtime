#include "Omni-Runtime/graph/event/event_manager.h"

namespace omni_runtime {
namespace graph {

EventManager::~EventManager() {
  static_cast<void>(Clear());
}

Status EventManager::Trigger(const std::string &key, const EventType type,
                             const EventAsyncStrategy strategy) {
  std::shared_ptr<Event> event;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = events_.find(key);
    OMNI_RETURN_VAL_IF_LOG(iter == events_.end(), NotFoundStatus("event not found: " + key), ERROR,
                      "event not found: " << key);
    event = iter->second;
  }
  return event->Process(type, strategy);
}

std::shared_future<void> EventManager::AsyncTrigger(const std::string &key,
                                                    const EventAsyncStrategy strategy) {
  std::shared_ptr<Event> event;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = events_.find(key);
    OMNI_RETURN_VAL_IF_LOG(iter == events_.end(), std::shared_future<void>(), ERROR,
                      "event not found: " << key);
    event = iter->second;
  }
  return event->AsyncProcess(strategy);
}

bool EventManager::SetContext(const std::shared_ptr<ParamManager> &param_manager,
                              const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool) {
  OMNI_RETURN_VAL_IF(param_manager == nullptr || thread_pool == nullptr, false);
  std::lock_guard<std::mutex> lock(mutex_);
  param_manager_ = param_manager;
  thread_pool_ = thread_pool;
  for (const auto &item : events_) {
    OMNI_RETURN_VAL_IF(!item.second->SetContext(param_manager, thread_pool), false);
  }
  return true;
}

Status EventManager::Reset() {
  std::vector<std::shared_ptr<Event>> events;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &item : events_) {
      events.emplace_back(item.second);
    }
  }
  for (const auto &event : events) {
    OMNI_RETURN_VAL_IF_LOG(!event->Wait(EventAsyncStrategy::kPipelineRunFinish),
                      InternalStatus("event wait failed"), ERROR, "event wait failed");
  }
  return Status();
}

Status EventManager::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  events_.clear();
  return Status();
}

Status EventManager::Init() {
  std::vector<std::shared_ptr<Event>> events;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    OMNI_RETURN_VAL_IF_LOG(param_manager_ == nullptr || thread_pool_ == nullptr,
                      InternalStatus("event context is unavailable"), ERROR,
                      "event context is unavailable");
    for (const auto &item : events_) {
      events.emplace_back(item.second);
    }
  }
  for (const auto &event : events) {
    OMNI_RETURN_STATUS_IF_NOT_OK(event->Init());
  }
  return Status();
}

Status EventManager::Destroy() {
  std::vector<std::shared_ptr<Event>> events;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &item : events_) {
      events.emplace_back(item.second);
    }
  }
  for (auto iter = events.rbegin(); iter != events.rend(); ++iter) {
    OMNI_RETURN_VAL_IF_LOG(!(*iter)->Wait(EventAsyncStrategy::kPipelineDestroy),
                      InternalStatus("event wait failed"), ERROR, "event wait failed");
    OMNI_RETURN_STATUS_IF_NOT_OK((*iter)->Destroy());
  }
  return Status();
}

} // namespace graph
} // namespace omni_runtime
