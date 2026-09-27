#pragma once

#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include "Omni-Runtime/utils/thread_pool.h"
#include "Omni-Runtime/utils/logger.h"

#include "Omni-Runtime/graph/basic/descriptor.h"
#include "Omni-Runtime/graph/basic/object.h"
#include "Omni-Runtime/graph/event/event_define.h"
#include "Omni-Runtime/graph/param/param_manager.h"
#include "Omni-Runtime/graph/param/passed_param.h"

namespace omni_runtime {
namespace graph {

class Storage;

class Event : public Object, public Descriptor {
public:
  Event() : Descriptor("event") {
  }
  ~Event() override = default;

protected:
  virtual bool Trigger(const EventParam *const param) = 0;

  template <typename T> std::shared_ptr<T> GetParam(const std::string &key) const {
    const auto manager = param_manager_.lock();
    OMNI_RETURN_VAL_IF_LOG(manager == nullptr, nullptr, ERROR, "param manager is unavailable");
    const auto param = manager->Get<T>(key);
    if (param != nullptr) {
      static_cast<void>(param->AddBacktrace(name()));
    }
    return param;
  }

private:
  Status Run() final {
    return UnimplementedStatus("Event cannot run directly");
  }

  Status Process(const EventType type, const EventAsyncStrategy strategy);
  std::shared_future<void> AsyncProcess(const EventAsyncStrategy strategy);
  bool Wait(const EventAsyncStrategy strategy);
  bool SetContext(const std::shared_ptr<ParamManager> &param_manager,
                  const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool);
  bool SetParam(const EventParam *const param);

  std::weak_ptr<ParamManager> param_manager_;
  std::shared_ptr<omni_runtime::utils::ThreadPool> thread_pool_;
  std::shared_ptr<EventParam> param_;
  std::mutex run_futures_mutex_;
  std::mutex destroy_futures_mutex_;
  std::vector<std::shared_future<void>> run_futures_;
  std::vector<std::shared_future<void>> destroy_futures_;

  friend class EventManager;
  friend class Storage;
};

} // namespace graph
} // namespace omni_runtime
