#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "Omni-Runtime/utils/logger.h"

#include "Omni-Runtime/graph/aspect/aspect_define.h"
#include "Omni-Runtime/graph/basic/descriptor.h"
#include "Omni-Runtime/graph/basic/object.h"
#include "Omni-Runtime/graph/event/event_manager.h"
#include "Omni-Runtime/graph/param/param_manager.h"
#include "Omni-Runtime/graph/param/passed_param.h"

namespace omni_runtime {
namespace graph {

class Storage;

class Aspect : public Object, public Descriptor {
public:
  Aspect() : Descriptor("aspect") {
  }
  virtual ~Aspect() = default;

  virtual Status BeginInit() {
    return Status();
  }

  virtual bool FinishInit(const Status &status) {
    static_cast<void>(status);
    return true;
  }

  virtual Status BeginRun() {
    return Status();
  }

  virtual bool FinishRun(const Status &status) {
    static_cast<void>(status);
    return true;
  }

  virtual Status BeginDestroy() {
    return Status();
  }

  virtual bool FinishDestroy(const Status &status) {
    static_cast<void>(status);
    return true;
  }

  virtual bool EnterCrashed() {
    return true;
  }

  virtual bool EnterTimeout() {
    return true;
  }

protected:
  template <typename T> std::shared_ptr<T> param() const {
    static_assert(std::is_base_of<AspectParam, T>::value,
                  "Aspect param must derive from AspectParam");
    return std::dynamic_pointer_cast<T>(param_);
  }

  template <typename T> std::shared_ptr<T> GetParam(const std::string &key) const {
    const auto manager = param_manager_.lock();
    OMNI_RETURN_VAL_IF_LOG(manager == nullptr, nullptr, ERROR, "param manager is unavailable");
    const auto value = manager->Get<T>(key);
    if (value != nullptr) {
      static_cast<void>(value->AddBacktrace(belong_name_));
    }
    return value;
  }

  Status Notify(const std::string &key, const EventType type,
                const EventAsyncStrategy strategy) const {
    const auto manager = event_manager_.lock();
    OMNI_RETURN_VAL_IF_LOG(manager == nullptr, InternalStatus("event manager is unavailable"), ERROR,
                      "event manager is unavailable");
    return manager->Trigger(key, type, strategy);
  }

private:
  Status Run() final {
    return UnimplementedStatus("Aspect cannot run directly");
  }

  bool SetParam(const AspectParam *const param) {
    if (param == nullptr) {
      param_.reset();
      return true;
    }
    param_ = param->Clone();
    return param_ != nullptr;
  }

  bool SetContext(const std::shared_ptr<ParamManager> &param_manager,
                  const std::shared_ptr<EventManager> &event_manager,
                  const std::string &belong_name) {
    OMNI_RETURN_VAL_IF(param_manager == nullptr || event_manager == nullptr, false);
    param_manager_ = param_manager;
    event_manager_ = event_manager;
    belong_name_ = belong_name;
    return true;
  }

  std::shared_ptr<AspectParam> param_;
  std::weak_ptr<ParamManager> param_manager_;
  std::weak_ptr<EventManager> event_manager_;
  std::string belong_name_;

  friend class AspectManager;
  friend class Storage;
};

template <typename... Args> class TemplateAspect : public Aspect {
public:
  explicit TemplateAspect(const Args &...args) : args_(args...) {
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
