#pragma once

#include <future>
#include <memory>
#include <vector>

#include "Omni-Runtime/utils/status_macros.h"
#include "Omni-Runtime/graph/element/element.h"
#include "Omni-Runtime/graph/element/element_manager.h"

namespace omni_runtime {
namespace graph {

class Group : public Element {
public:
  explicit Group(const ElementType element_type);
  ~Group() override = default;

  Status AddElement(const std::shared_ptr<Element> &element);

protected:
  Status Init() override;
  Status Destroy() override;
  Status WaitPendingTasks() override;
  bool IsSerializable() const override;
  std::vector<std::shared_ptr<Element>> ChildElements() const override;
  virtual Status AddElementExtra(const std::shared_ptr<Element> &element) {
    static_cast<void>(element);
    return Status();
  }
  bool
  SetContextExtra(const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool) override;

  const std::vector<std::shared_ptr<Element>> &children() const {
    return children_;
  }

private:
  std::vector<std::shared_ptr<Element>> children_;

  friend class Region;
  friend class Mutable;
};

class Cluster final : public Group {
public:
  Cluster();
  ~Cluster() override = default;

private:
  Status Run() final;
};

class Condition : public Group {
public:
  Condition();
  ~Condition() override = default;

protected:
  virtual Index Choose() = 0;

private:
  Status Run() final;
};

class Region final : public Group {
public:
  Region();
  ~Region() override = default;

  bool set_engine_type(const EngineType engine_type);
  std::size_t Trim();

private:
  Status Init() final;
  Status Run() final;
  Status Destroy() final;
  Status AddElementExtra(const std::shared_ptr<Element> &element) final;
  bool SetContextExtra(const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool) final;
  bool IsSerializable() const final;

  std::shared_ptr<ElementManager> manager_;
};

class Mutable : public Group {
public:
  Mutable();
  ~Mutable() override = default;

protected:
  virtual Status Reshape(std::vector<std::shared_ptr<Element>> *const elements) = 0;

private:
  Status Run() final;
  bool IsSerializable() const final {
    return false;
  }
};

class Some : public Group {
public:
  Some();
  ~Some() override;

protected:
  virtual std::size_t Threshold() = 0;

private:
  Status Run() final;
  Status WaitPendingTasks() final;
  Status AddElementExtra(const std::shared_ptr<Element> &element) final;
  bool IsSerializable() const final {
    return false;
  }

  std::mutex pending_mutex_;
  std::vector<std::shared_future<Status>> pending_futures_;
};

template <MultiConditionType type> class MultiCondition final : public Group {
public:
  MultiCondition() : Group(ElementType::kMultiCondition) {
  }
  ~MultiCondition() override = default;

private:
  Status Run() final {
    std::vector<std::shared_ptr<Element>> matched;
    for (const auto &child : children()) {
      if (child->IsMatch()) {
        matched.emplace_back(child);
      }
    }
    if (type == MultiConditionType::kSerial || matched.size() <= 1U) {
      for (const auto &child : matched) {
        OMNI_RETURN_STATUS_IF_NOT_OK(child->Process(Element::Lifecycle::kRun));
      }
      return Status();
    }
    const auto pool = thread_pool();
    OMNI_RETURN_VAL_IF_LOG(pool == nullptr, InternalStatus("thread pool is unavailable"), ERROR,
                      "thread pool is unavailable");
    std::vector<std::future<Status>> futures;
    for (const auto &child : matched) {
      futures.emplace_back(
          pool->SafePost([child]() { return child->Process(Element::Lifecycle::kRun); }));
    }
    Status first_error;
    for (auto &future : futures) {
      const Status status = future.get();
      if (first_error.ok() && !status.ok()) {
        first_error = status;
      }
    }
    return first_error;
  }

  bool IsSerializable() const final {
    return type == MultiConditionType::kSerial && Group::IsSerializable();
  }
};

} // namespace graph
} // namespace omni_runtime
