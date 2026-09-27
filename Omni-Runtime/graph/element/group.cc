#include "Omni-Runtime/graph/element/group.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <thread>

#include "Omni-Runtime/utils/status_macros.h"

namespace omni_runtime {
namespace graph {

Group::Group(const ElementType element_type) : Element(element_type) {
}

Status Group::AddElement(const std::shared_ptr<Element> &element) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr, InvalidArgumentStatus("group element is null"), ERROR,
                    "group element is null");
  OMNI_RETURN_VAL_IF_LOG(IsInitialized(), InvalidArgumentStatus("group is initialized"), ERROR,
                    "group is initialized");
  OMNI_RETURN_VAL_IF_LOG(std::find(children_.begin(), children_.end(), element) != children_.end(),
                    InvalidArgumentStatus("group element is duplicated"), ERROR,
                    "group element is duplicated");
  OMNI_RETURN_VAL_IF_LOG(!element->SetBelong(shared_from_this()),
                    InternalStatus("failed to set group owner"), ERROR,
                    "failed to set group owner");
  children_.emplace_back(element);
  return AddElementExtra(element);
}

Status Group::Init() {
  for (const auto &child : children_) {
    OMNI_RETURN_STATUS_IF_NOT_OK(child->Process(Element::Lifecycle::kInit));
  }
  return Status();
}

Status Group::Destroy() {
  Status first_error;
  for (auto iter = children_.rbegin(); iter != children_.rend(); ++iter) {
    const Status status = (*iter)->Process(Element::Lifecycle::kDestroy);
    if (first_error.ok() && !status.ok()) {
      first_error = status;
    }
  }
  return first_error;
}

Status Group::WaitPendingTasks() {
  Status first_error = Element::WaitPendingTasks();
  for (const auto &child : children_) {
    const Status status = child->WaitPendingTasks();
    if (first_error.ok() && !status.ok()) {
      first_error = status;
    }
  }
  return first_error;
}

bool Group::IsSerializable() const {
  return std::all_of(
      children_.begin(), children_.end(),
      [](const std::shared_ptr<Element> &element) { return element->IsSerializable(); });
}

std::vector<std::shared_ptr<Element>> Group::ChildElements() const {
  return children_;
}

bool Group::SetContextExtra(const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool) {
  OMNI_RETURN_VAL_IF(thread_pool == nullptr, false);
  for (const auto &child : children_) {
    OMNI_RETURN_VAL_IF(!child->SetContext(param_manager_, event_manager_, stage_manager_, thread_pool),
                  false);
  }
  return true;
}

Cluster::Cluster() : Group(ElementType::kCluster) {
}

Status Cluster::Run() {
  for (const auto &child : children()) {
    OMNI_RETURN_STATUS_IF_NOT_OK(child->Process(Element::Lifecycle::kRun));
  }
  return Status();
}

Condition::Condition() : Group(ElementType::kCondition) {
}

Status Condition::Run() {
  const Index index = Choose();
  if (index == -1 && !children().empty()) {
    return children().back()->Process(Element::Lifecycle::kRun);
  }
  if (index >= 0 && static_cast<std::size_t>(index) < children().size()) {
    return children()[static_cast<std::size_t>(index)]->Process(Element::Lifecycle::kRun);
  }
  return Status();
}

Region::Region() : Group(ElementType::kRegion), manager_(std::make_shared<ElementManager>()) {
}

bool Region::set_engine_type(const EngineType engine_type) {
  return manager_->set_engine_type(engine_type);
}

std::size_t Region::Trim() {
  return manager_->Trim();
}

Status Region::Init() {
  return manager_->Init();
}

Status Region::Run() {
  return manager_->Run();
}

Status Region::Destroy() {
  return manager_->Destroy();
}

Status Region::AddElementExtra(const std::shared_ptr<Element> &element) {
  return manager_->Add(element);
}

bool Region::SetContextExtra(const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool) {
  OMNI_RETURN_VAL_IF(!Group::SetContextExtra(thread_pool), false);
  return manager_->SetThreadPool(thread_pool);
}

bool Region::IsSerializable() const {
  return manager_->IsSerializable();
}

Mutable::Mutable() : Group(ElementType::kMutable) {
}

Status Mutable::Run() {
  std::vector<std::shared_ptr<Element>> elements = children();
  for (const auto &element : elements) {
    element->loop_count_ = kDefaultLoopCount;
    element->is_visible_ = false;
    OMNI_RETURN_VAL_IF_LOG(!element->ClearRelations(), InternalStatus("mutable element reset failed"),
                      ERROR, "mutable element reset failed");
  }
  OMNI_RETURN_STATUS_IF_NOT_OK(Reshape(&elements));
  ElementManager manager;
  OMNI_RETURN_VAL_IF_LOG(!manager.SetThreadPool(thread_pool()),
                    InternalStatus("mutable thread pool setup failed"), ERROR,
                    "mutable thread pool setup failed");
  for (const auto &element : elements) {
    if (element->is_visible_) {
      OMNI_RETURN_STATUS_IF_NOT_OK(manager.Add(element));
    }
  }
  OMNI_RETURN_STATUS_IF_NOT_OK(manager.Validate());
  return manager.Run();
}

Some::Some() : Group(ElementType::kSome) {
}

Some::~Some() {
  static_cast<void>(WaitPendingTasks());
}

Status Some::Run() {
  const std::size_t threshold = Threshold();
  OMNI_RETURN_VAL_IF_LOG(threshold == 0U || threshold > children().size(),
                    InvalidArgumentStatus("some threshold is invalid"), ERROR,
                    "some threshold is invalid");
  const auto pool = thread_pool();
  OMNI_RETURN_VAL_IF_LOG(pool == nullptr, InternalStatus("thread pool is unavailable"), ERROR,
                    "thread pool is unavailable");
  std::vector<std::shared_future<Status>> futures;
  for (const auto &child : children()) {
    futures.emplace_back(
        pool->SafePost([child]() { return child->Process(Element::Lifecycle::kRun); }).share());
  }
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_futures_ = futures;
  }
  Status first_error;
  std::size_t completed_count = 0U;
  std::vector<bool> is_completed(futures.size(), false);
  while (completed_count < threshold) {
    bool has_progress = false;
    for (std::size_t future_index = 0U; future_index < futures.size(); ++future_index) {
      auto &future = futures[future_index];
      if (is_completed[future_index] || !future.valid() ||
          future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        continue;
      }
      const Status status = future.get();
      is_completed[future_index] = true;
      ++completed_count;
      has_progress = true;
      if (first_error.ok() && !status.ok()) {
        first_error = status;
      }
    }
    if (!has_progress) {
      std::this_thread::yield();
    }
  }
  return first_error;
}

Status Some::WaitPendingTasks() {
  std::vector<std::shared_future<Status>> futures;
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    futures.swap(pending_futures_);
  }
  Status first_error;
  for (const auto &future : futures) {
    if (!future.valid()) {
      continue;
    }
    const Status status = future.get();
    if (first_error.ok() && !status.ok()) {
      first_error = status;
    }
  }
  const Status group_status = Group::WaitPendingTasks();
  return first_error.ok() ? group_status : first_error;
}

Status Some::AddElementExtra(const std::shared_ptr<Element> &element) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr, InvalidArgumentStatus("some element is null"), ERROR,
                    "some element is null");
  if (element->timeout_ms() == kNoTimeoutMs) {
    OMNI_RETURN_VAL_IF_LOG(
        !element->set_timeout(kMaxBlockingTimeoutMs, ElementTimeoutStrategy::kHoldByPipeline),
        InvalidArgumentStatus("some element timeout setup failed"), ERROR,
        "some element timeout setup failed");
  }
  return Status();
}

} // namespace graph
} // namespace omni_runtime
