#pragma once

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "Omni-Runtime/utils/status_macros.h"
#include "Omni-Runtime/graph/event/event.h"

namespace omni_runtime {
namespace graph {

class Storage;

class EventManager final : public Object {
public:
  EventManager() = default;
  ~EventManager() override;

  template <typename TEvent, typename TParam>
  Status Create(const std::string &key, const TParam *const param) {
    static_assert(std::is_base_of<Event, TEvent>::value, "TEvent must inherit Event");
    static_assert(std::is_base_of<EventParam, TParam>::value, "TParam must inherit EventParam");
    OMNI_RETURN_VAL_IF_LOG(key.empty(), InvalidArgumentStatus("event key is empty"), ERROR,
                      "event key is empty");
    std::lock_guard<std::mutex> lock(mutex_);
    OMNI_RETURN_VAL_IF_LOG(events_.find(key) != events_.end(),
                      InvalidArgumentStatus("event key is duplicated: " + key), ERROR,
                      "event key is duplicated: " << key);
    auto event = std::make_shared<TEvent>();
    OMNI_RETURN_VAL_IF_LOG(!event->SetParam(param), InternalStatus("event param clone failed"), ERROR,
                      "event param clone failed");
    if (param_manager_ != nullptr && thread_pool_ != nullptr) {
      OMNI_RETURN_VAL_IF_LOG(!event->SetContext(param_manager_, thread_pool_),
                        InternalStatus("event context setup failed"), ERROR,
                        "event context setup failed");
    }
    events_.emplace(key, std::move(event));
    return Status();
  }

  Status Trigger(const std::string &key, const EventType type, const EventAsyncStrategy strategy);
  std::shared_future<void> AsyncTrigger(const std::string &key, const EventAsyncStrategy strategy);
  bool SetContext(const std::shared_ptr<ParamManager> &param_manager,
                  const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool);
  Status Reset();
  Status Clear();

  Status Init() final;
  Status Destroy() final;

private:
  Status Run() final {
    return UnimplementedStatus("EventManager cannot run");
  }

  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Event>> events_;
  std::shared_ptr<ParamManager> param_manager_;
  std::shared_ptr<omni_runtime::utils::ThreadPool> thread_pool_;

  friend class Storage;
};

} // namespace graph
} // namespace omni_runtime
