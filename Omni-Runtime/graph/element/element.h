#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/utils/thread_pool.h"
#include "Omni-Runtime/utils/logger.h"

#include "Omni-Runtime/graph/aspect/aspect_manager.h"
#include "Omni-Runtime/graph/basic/descriptor.h"
#include "Omni-Runtime/graph/basic/object.h"
#include "Omni-Runtime/graph/element/element_define.h"
#include "Omni-Runtime/graph/event/event_manager.h"
#include "Omni-Runtime/graph/param/param_manager.h"
#include "Omni-Runtime/graph/pipeline/perf_define.h"
#include "Omni-Runtime/graph/stage/stage_manager.h"

namespace omni_runtime {
namespace graph {

class Storage;

class Element : public Object, public Descriptor, public std::enable_shared_from_this<Element> {
public:
  explicit Element(const ElementType element_type);
  ~Element() override;

  bool set_name(const std::string &name);
  bool set_loop_count(const std::size_t loop_count);
  bool set_level(const Level level);
  bool set_visible(const bool is_visible);
  bool set_timeout(const Milliseconds timeout_ms, const ElementTimeoutStrategy timeout_strategy);
  bool set_binding_index(const Index binding_index);
  bool set_macro(const bool is_macro);

  std::size_t loop_count() const {
    return loop_count_;
  }

  Level level() const {
    return level_;
  }

  Milliseconds timeout_ms() const {
    return timeout_ms_;
  }

  Index binding_index() const {
    return binding_index_;
  }

  ElementState state() const;
  ElementType element_type() const {
    return element_type_;
  }

  bool IsGroup() const;
  bool IsAdapter() const;
  bool IsNode() const;
  bool IsTimeout() const;
  bool IsInitialized() const {
    return is_initialized_;
  }

  bool is_visible() const {
    return is_visible_;
  }

  bool is_macro() const {
    return is_macro_;
  }

  ElementTimeoutStrategy timeout_strategy() const {
    return timeout_strategy_;
  }

  PerfInfo perf_info() const;

  Status AddDependency(const std::shared_ptr<Element> &element);
  Status RemoveDependency(const std::shared_ptr<Element> &element);
  Status GetRelation(ElementRelation *const relation) const;
  Status AddAspect(const std::shared_ptr<Aspect> &aspect, const AspectParam *const param);
  Status PopLastAspect();

  template <typename T> Status AddLocalParam(const std::string &key, const T *const param) {
    static_assert(std::is_base_of<ElementParam, T>::value,
                  "Element param must derive from ElementParam");
    OMNI_RETURN_VAL_IF_LOG(param == nullptr, InvalidArgumentStatus("element param is null"), ERROR,
                      "element param is null");
    OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("element is initialized"), ERROR,
                      "element is initialized");
    const auto cloned = std::dynamic_pointer_cast<T>(param->Clone());
    OMNI_RETURN_VAL_IF_LOG(cloned == nullptr, InternalStatus("element param clone failed"), ERROR,
                      "element param clone failed");
    std::lock_guard<std::mutex> lock(local_param_mutex_);
    local_params_[key] = cloned;
    return Status();
  }

protected:
  Status Init() override {
    return Status();
  }

  Status Destroy() override {
    return Status();
  }

  virtual Status PrepareRun() {
    return Status();
  }

  virtual Status CheckRunResult() {
    return Status();
  }

  virtual bool IsHold() {
    return false;
  }

  virtual bool IsMatch() {
    return false;
  }

  virtual bool IsSerializable() const {
    return true;
  }

  virtual Status WaitPendingTasks() {
    return WaitAsync();
  }

  template <typename T> std::shared_ptr<T> GetParam(const std::string &key) const {
    OMNI_RETURN_VAL_IF_LOG(param_manager_ == nullptr, nullptr, ERROR, "param manager is unavailable");
    const auto param = param_manager_->Get<T>(key);
    if (param != nullptr) {
      static_cast<void>(param->AddBacktrace(name()));
    }
    return param;
  }

  template <typename T> std::shared_ptr<T> GetLocalParam(const std::string &key) const {
    std::lock_guard<std::mutex> lock(local_param_mutex_);
    const auto iter = local_params_.find(key);
    if (iter == local_params_.end()) {
      return nullptr;
    }
    return std::dynamic_pointer_cast<T>(iter->second);
  }

  Status Notify(const std::string &key, const EventType type,
                const EventAsyncStrategy strategy) const;
  Status EnterStage(const std::string &key) const;
  Status Spawn(const TaskGroup &tasks, const Milliseconds timeout_ms) const;
  std::shared_ptr<omni_runtime::utils::ThreadPool> thread_pool() const {
    return thread_pool_;
  }

  virtual std::vector<std::shared_ptr<Element>> ChildElements() const {
    return {};
  }

private:
  enum class Lifecycle : uint8_t {
    kInit = 0,
    kRun = 1,
    kDestroy = 2,
  };

  Status Process(const Lifecycle lifecycle);
  Status ProcessRun();
  Status RunWithTimeout();
  Status WaitAsync();
  Status Reflect(const AspectType type, const Status &status) const;
  bool SetContext(const std::shared_ptr<ParamManager> &param_manager,
                  const std::shared_ptr<EventManager> &event_manager,
                  const std::shared_ptr<StageManager> &stage_manager,
                  const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool);
  virtual bool
  SetContextExtra(const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool) {
    static_cast<void>(thread_pool);
    return true;
  }
  bool SetState(const ElementState state);
  bool WaitIfSuspended();
  bool AddSuccessor(const std::shared_ptr<Element> &element);
  bool RemoveSuccessor(const std::shared_ptr<Element> &element);
  bool SetBelong(const std::shared_ptr<Element> &belong);
  bool ClearRelations();
  bool IsRegistered() const;
  bool HasPathTo(const std::shared_ptr<Element> &target) const;
  Status Dump(std::ostream *const stream) const;
  bool RecordRunTime(const double elapsed_time_ms);

  ElementType element_type_ = ElementType::kElement;
  std::atomic<ElementState> state_ = ElementState::kNormal;
  std::atomic<bool> is_done_ = false;
  bool is_visible_ = true;
  bool is_initialized_ = false;
  bool is_prepared_ = false;
  bool is_macro_ = false;
  std::size_t loop_count_ = kDefaultLoopCount;
  Level level_ = kDefaultElementLevel;
  Milliseconds timeout_ms_ = kNoTimeoutMs;
  ElementTimeoutStrategy timeout_strategy_ = ElementTimeoutStrategy::kAsError;
  Index binding_index_ = -1;

  mutable std::mutex relation_mutex_;
  std::vector<std::weak_ptr<Element>> dependencies_;
  std::vector<std::weak_ptr<Element>> successors_;
  std::weak_ptr<Element> belong_;

  mutable std::mutex local_param_mutex_;
  std::unordered_map<std::string, std::shared_ptr<ElementParam>> local_params_;
  std::shared_ptr<AspectManager> aspect_manager_;
  std::shared_ptr<ParamManager> param_manager_;
  std::shared_ptr<EventManager> event_manager_;
  std::shared_ptr<StageManager> stage_manager_;
  std::shared_ptr<omni_runtime::utils::ThreadPool> thread_pool_;
  std::shared_future<Status> async_result_;
  mutable std::mutex async_mutex_;
  std::mutex suspend_mutex_;
  std::condition_variable suspend_condition_;
  mutable std::mutex perf_mutex_;
  PerfInfo perf_info_;

  friend class ElementManager;
  friend class Pipeline;
  friend class Storage;
  friend class Group;
  friend class Cluster;
  friend class Condition;
  friend class Region;
  friend class Mutable;
  friend class Some;
  friend class Fence;
  template <MultiConditionType type> friend class MultiCondition;
  template <typename T> friend class Singleton;
};

} // namespace graph
} // namespace omni_runtime
