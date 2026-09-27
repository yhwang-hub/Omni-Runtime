#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#include "Omni-Runtime/utils/logger.h"

#include "Omni-Runtime/graph/basic/descriptor.h"
#include "Omni-Runtime/graph/basic/object.h"
#include "Omni-Runtime/graph/basic/types.h"
#include "Omni-Runtime/graph/event/event_manager.h"
#include "Omni-Runtime/graph/param/param_manager.h"
#include "Omni-Runtime/graph/param/passed_param.h"

namespace omni_runtime {
namespace graph {

class Storage;

class Daemon : public Object, public Descriptor {
public:
  Daemon() : Descriptor("daemon") {
  }
  ~Daemon() override;

  Milliseconds interval_ms() const;
  bool set_interval_ms(const Milliseconds interval_ms);
  bool SetParam(const DaemonParam *const param);

protected:
  virtual bool DaemonTask(const DaemonParam *const param) = 0;

  virtual Milliseconds ModifyIntervalMs(const DaemonParam *const param) {
    static_cast<void>(param);
    return 0;
  }

  template <typename T> std::shared_ptr<T> GetParam(const std::string &key) const {
    const auto manager = param_manager_.lock();
    OMNI_RETURN_VAL_IF_LOG(manager == nullptr, nullptr, ERROR, "param manager is unavailable");
    const auto param = manager->Get<T>(key);
    if (param != nullptr) {
      static_cast<void>(param->AddBacktrace(name()));
    }
    return param;
  }

  Status Notify(const std::string &key, const EventType type,
                const EventAsyncStrategy strategy) const {
    const auto manager = event_manager_.lock();
    OMNI_RETURN_VAL_IF_LOG(manager == nullptr, InternalStatus("event manager is unavailable"), ERROR,
                      "event manager is unavailable");
    return manager->Trigger(key, type, strategy);
  }

private:
  Status Init() final;
  Status Run() final {
    return UnimplementedStatus("Daemon cannot run directly");
  }
  Status Destroy() final;
  bool SetContext(const std::shared_ptr<ParamManager> &param_manager,
                  const std::shared_ptr<EventManager> &event_manager);
  void WorkLoop();

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::thread worker_;
  std::weak_ptr<ParamManager> param_manager_;
  std::weak_ptr<EventManager> event_manager_;
  std::shared_ptr<DaemonParam> param_;
  Milliseconds interval_ms_ = 0;
  bool is_running_ = false;

  friend class DaemonManager;
  friend class Storage;
};

template <typename... Args> class TemplateDaemon : public Daemon {
public:
  explicit TemplateDaemon(const Args &...args) : args_(args...) {
  }

protected:
  std::tuple<Args...> &args() {
    return args_;
  }

private:
  std::tuple<Args...> args_;
};

} // namespace graph
} // namespace omni_runtime
