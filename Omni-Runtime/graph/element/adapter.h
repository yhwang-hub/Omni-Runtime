#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <set>

#include "Omni-Runtime/graph/element/element.h"

namespace omni_runtime {
namespace graph {

class Adapter : public Element {
public:
  explicit Adapter(const ElementType element_type) : Element(element_type) {
  }
  ~Adapter() override = default;
};

class Function final : public Adapter {
public:
  Function();
  ~Function() override = default;

  bool SetFunction(const FunctionType type, const std::function<Status()> &function);

private:
  Status Init() final;
  Status Run() final;
  Status Destroy() final;

  std::function<Status()> init_function_;
  std::function<Status()> run_function_;
  std::function<Status()> destroy_function_;
};

class Fence final : public Adapter {
public:
  Fence();
  ~Fence() override = default;

  Status WaitElement(const std::shared_ptr<Element> &element);
  Status WaitElements(const std::vector<std::shared_ptr<Element>> &elements);
  bool Clear();

private:
  Status Run() final;

  std::mutex mutex_;
  std::vector<std::weak_ptr<Element>> elements_;
};

template <int32_t thread_delta> class Coordinator final : public Adapter {
public:
  Coordinator() : Adapter(ElementType::kCoordinator) {
  }
  ~Coordinator() override = default;

private:
  Status Run() final {
    LOG_INFO("Coordinator requested thread delta "
             << thread_delta << "; thread pool size is immutable after construction");
    return Status();
  }
};

template <typename T> class Singleton final : public Adapter {
public:
  Singleton() : Adapter(ElementType::kSingleton) {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (instance_ == nullptr) {
      instance_ = std::make_shared<T>();
    }
  }
  ~Singleton() override = default;

private:
  Status Init() final {
    bool was_initialized = false;
    if (!is_initialized_.compare_exchange_strong(was_initialized, true)) {
      return Status();
    }
    const Status status = instance_->Process(Element::Lifecycle::kInit);
    if (!status.ok()) {
      is_initialized_.store(false, std::memory_order_release);
    }
    return status;
  }

  Status Run() final {
    return instance_->Process(Element::Lifecycle::kRun);
  }

  Status Destroy() final {
    bool was_initialized = true;
    if (!is_initialized_.compare_exchange_strong(was_initialized, false)) {
      return Status();
    }
    return instance_->Process(Element::Lifecycle::kDestroy);
  }

  bool SetContextExtra(const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool) final {
    OMNI_RETURN_VAL_IF(thread_pool == nullptr || instance_ == nullptr, false);
    return instance_->SetContext(param_manager_, event_manager_, stage_manager_, thread_pool);
  }

  inline static std::mutex instance_mutex_;
  inline static std::shared_ptr<T> instance_;
  inline static std::atomic<bool> is_initialized_ = false;
};

} // namespace graph
} // namespace omni_runtime
