#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "Omni-Runtime/utils/thread_pool.h"
#include "Omni-Runtime/utils/logger.h"

#include "Omni-Runtime/utils/status_macros.h"
#include "Omni-Runtime/graph/basic/descriptor.h"
#include "Omni-Runtime/graph/daemon/daemon_manager.h"
#include "Omni-Runtime/graph/element/element_include.h"
#include "Omni-Runtime/graph/event/event_manager.h"
#include "Omni-Runtime/graph/param/param_manager.h"
#include "Omni-Runtime/graph/pipeline/pipeline_define.h"
#include "Omni-Runtime/graph/stage/stage_manager.h"

namespace omni_runtime {
namespace graph {

class Storage;

class Pipeline final : public Object, public Descriptor {
public:
  Pipeline();
  ~Pipeline() override;

  Status Init() final;
  Status Run() final;
  Status Destroy() final;
  Status Process(const std::size_t run_count);
  std::future<Status> AsyncRun(const std::launch policy);
  std::future<Status> AsyncProcess(const std::size_t run_count, const std::launch policy);

  bool set_thread_count(const uint32_t thread_count);
  bool set_engine_type(const EngineType engine_type);
  PipelineState state() const {
    return state_.load(std::memory_order_acquire);
  }

  Status Cancel();
  Status Suspend();
  Status Resume();
  Status Dump(std::ostream *const stream) const;
  Status Perf(std::ostream *const stream) const;
  Status Save(const std::string &path) const;
  Status Load(const std::string &path);
  std::size_t MaxParallelism() const;
  std::size_t Trim();
  Status MakeSerial();
  bool AreSeparated(const std::shared_ptr<Element> &first,
                    const std::shared_ptr<Element> &second) const;

  template <typename T> Status CreateParam(const std::string &key, const bool enable_backtrace) {
    OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("pipeline is initialized"), ERROR,
                      "pipeline is initialized");
    return param_manager_->Create<T>(key, enable_backtrace);
  }

  template <typename T> std::shared_ptr<T> GetParam(const std::string &key) const {
    return param_manager_->Get<T>(key);
  }

  Status RemoveParam(const std::string &key) {
    OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("pipeline is initialized"), ERROR,
                      "pipeline is initialized");
    return param_manager_->Remove(key);
  }

  Status RegisterElement(const std::shared_ptr<Element> &element,
                         const std::vector<std::shared_ptr<Element>> &dependencies,
                         const std::string &name, const std::size_t loop_count);

  template <typename TNode, typename... Args>
  std::shared_ptr<TNode> CreateNode(const NodeInfo &info, Args &&...args) {
    static_assert(std::is_base_of<Node, TNode>::value, "TNode must derive from Node");
    OMNI_RETURN_VAL_IF_LOG(is_initialized_, nullptr, ERROR, "pipeline is initialized");
    auto node = std::make_shared<TNode>(std::forward<Args>(args)...);
    OMNI_RETURN_VAL_IF_LOG(!node->set_name(info.name) || !node->set_loop_count(info.loop_count), nullptr,
                      ERROR, "node configuration failed");
    for (const auto &dependency : info.dependencies) {
      OMNI_RETURN_VAL_IF_LOG(!node->AddDependency(dependency).ok(), nullptr, ERROR,
                        "node dependency setup failed");
    }
    return node;
  }

  template <typename TNode, typename... Args>
  std::shared_ptr<TNode> RegisterNode(const NodeInfo &info, Args &&...args) {
    auto node = std::make_shared<TNode>(std::forward<Args>(args)...);
    OMNI_RETURN_VAL_IF_LOG(!RegisterElement(node, info.dependencies, info.name, info.loop_count).ok(),
                      nullptr, ERROR, "node registration failed");
    return node;
  }

  template <typename TGroup>
  std::shared_ptr<TGroup> CreateGroup(const std::vector<std::shared_ptr<Element>> &elements,
                                      const std::vector<std::shared_ptr<Element>> &dependencies,
                                      const std::string &name, const std::size_t loop_count) {
    static_assert(std::is_base_of<Group, TGroup>::value, "TGroup must derive from Group");
    OMNI_RETURN_VAL_IF_LOG(is_initialized_, nullptr, ERROR, "pipeline is initialized");
    auto group = std::make_shared<TGroup>();
    for (const auto &element : elements) {
      OMNI_RETURN_VAL_IF_LOG(!group->AddElement(element).ok(), nullptr, ERROR,
                        "group element setup failed");
    }
    OMNI_RETURN_VAL_IF_LOG(!group->set_name(name) || !group->set_loop_count(loop_count), nullptr, ERROR,
                      "group configuration failed");
    for (const auto &dependency : dependencies) {
      OMNI_RETURN_VAL_IF_LOG(!group->AddDependency(dependency).ok(), nullptr, ERROR,
                        "group dependency setup failed");
    }
    return group;
  }

  template <typename TAspect, typename TParam>
  Status AddAspect(const std::vector<std::shared_ptr<Element>> &elements,
                   const TParam *const param) {
    static_assert(std::is_base_of<Aspect, TAspect>::value, "TAspect must derive from Aspect");
    static_assert(std::is_base_of<AspectParam, TParam>::value,
                  "TParam must derive from AspectParam");
    OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("pipeline is initialized"), ERROR,
                      "pipeline is initialized");
    std::vector<std::shared_ptr<Element>> targets = elements;
    if (targets.empty()) {
      OMNI_RETURN_STATUS_IF_NOT_OK(GetGlobalAspectTargets(&targets));
    } else {
      for (const auto &element : targets) {
        OMNI_RETURN_VAL_IF_LOG(!element_manager_->Find(element),
                          InvalidArgumentStatus("aspect element is not registered"), ERROR,
                          "aspect element is not registered");
      }
    }
    for (const auto &element : targets) {
      OMNI_RETURN_STATUS_IF_NOT_OK(element->AddAspect(std::make_shared<TAspect>(), param));
    }
    return Status();
  }

  template <typename TDaemon, typename TParam>
  Status AddDaemon(const Milliseconds interval_ms, const TParam *const param) {
    static_assert(std::is_base_of<Daemon, TDaemon>::value, "TDaemon must derive from Daemon");
    static_assert(std::is_base_of<DaemonParam, TParam>::value,
                  "TParam must derive from DaemonParam");
    OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("pipeline is initialized"), ERROR,
                      "pipeline is initialized");
    auto daemon = std::make_shared<TDaemon>();
    OMNI_RETURN_VAL_IF_LOG(!daemon->set_interval_ms(interval_ms) || !daemon->SetParam(param),
                      InvalidArgumentStatus("daemon configuration failed"), ERROR,
                      "daemon configuration failed");
    return daemon_manager_->Add(daemon);
  }

  template <typename TEvent, typename TParam>
  Status AddEvent(const std::string &key, const TParam *const param) {
    OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("pipeline is initialized"), ERROR,
                      "pipeline is initialized");
    return event_manager_->Create<TEvent, TParam>(key, param);
  }

  template <typename TStage, typename TParam>
  Status AddStage(const std::string &key, const int32_t threshold, const TParam *const param) {
    OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("pipeline is initialized"), ERROR,
                      "pipeline is initialized");
    return stage_manager_->Create<TStage, TParam>(key, threshold, param);
  }

private:
  Status SetState(const PipelineState state);
  Status ConfigureElements();
  Status GetGlobalAspectTargets(std::vector<std::shared_ptr<Element>> *const targets) const;

  std::shared_ptr<ElementManager> element_manager_;
  std::shared_ptr<ParamManager> param_manager_;
  std::shared_ptr<DaemonManager> daemon_manager_;
  std::shared_ptr<EventManager> event_manager_;
  std::shared_ptr<StageManager> stage_manager_;
  std::shared_ptr<omni_runtime::utils::ThreadPool> thread_pool_;
  PipelineConfig config_;
  std::atomic<PipelineState> state_ = PipelineState::kNormal;
  bool is_initialized_ = false;

  friend class Storage;
};

} // namespace graph
} // namespace omni_runtime
