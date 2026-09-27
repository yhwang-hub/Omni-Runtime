#include "Omni-Runtime/graph/element/element_manager.h"

#include <algorithm>
#include <future>
#include <queue>
#include <unordered_set>

#include "Omni-Runtime/utils/status_macros.h"

namespace omni_runtime {
namespace graph {

ElementManager::~ElementManager() {
  static_cast<void>(Clear());
}

Status ElementManager::Add(const std::shared_ptr<Element> &element) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr, InvalidArgumentStatus("element is null"), ERROR,
                    "element is null");
  std::lock_guard<std::mutex> lock(mutex_);
  OMNI_RETURN_VAL_IF_LOG(std::find(elements_.begin(), elements_.end(), element) != elements_.end(),
                    InvalidArgumentStatus("element is duplicated"), ERROR,
                    "element is duplicated: " << element->name());
  elements_.emplace_back(element);
  return Status();
}

Status ElementManager::Remove(const std::shared_ptr<Element> &element) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr, InvalidArgumentStatus("element is null"), ERROR,
                    "element is null");
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = std::find(elements_.begin(), elements_.end(), element);
  OMNI_RETURN_VAL_IF_LOG(iter == elements_.end(), NotFoundStatus("element not found"), ERROR,
                    "element not found");
  elements_.erase(iter);
  return Status();
}

bool ElementManager::Find(const std::shared_ptr<Element> &element) const {
  OMNI_RETURN_VAL_IF(element == nullptr, false);
  std::lock_guard<std::mutex> lock(mutex_);
  return std::find(elements_.begin(), elements_.end(), element) != elements_.end();
}

Status ElementManager::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  elements_.clear();
  return Status();
}

bool ElementManager::set_engine_type(const EngineType engine_type) {
  engine_type_ = engine_type;
  return true;
}

bool ElementManager::SetThreadPool(
    const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool) {
  OMNI_RETURN_VAL_IF(thread_pool == nullptr, false);
  thread_pool_ = thread_pool;
  return true;
}

Status ElementManager::GetElements(std::vector<std::shared_ptr<Element>> *const elements) const {
  OMNI_RETURN_VAL_IF_LOG(elements == nullptr, InvalidArgumentStatus("elements is null"), ERROR,
                    "elements is null");
  std::lock_guard<std::mutex> lock(mutex_);
  *elements = elements_;
  return Status();
}

Status ElementManager::SetState(const ElementState state) {
  std::vector<std::shared_ptr<Element>> elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(GetElements(&elements));
  for (const auto &element : elements) {
    OMNI_RETURN_VAL_IF_LOG(!element->SetState(state), InternalStatus("element state update failed"),
                      ERROR, "element state update failed");
  }
  return Status();
}

Status ElementManager::WaitAsyncElements() {
  std::vector<std::shared_ptr<Element>> elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(GetElements(&elements));
  Status first_error;
  for (const auto &element : elements) {
    const Status status = element->WaitPendingTasks();
    if (first_error.ok() && !status.ok() &&
        element->timeout_strategy_ == ElementTimeoutStrategy::kHoldByPipeline) {
      first_error = status;
    }
    if (element->state_.load(std::memory_order_acquire) == ElementState::kTimeout) {
      element->SetState(ElementState::kNormal);
    }
  }
  return first_error;
}

Status ElementManager::Validate() const {
  std::vector<std::vector<std::shared_ptr<Element>>> batches;
  return GetTopologicalBatches(&batches);
}

std::size_t ElementManager::MaxParallelism() const {
  std::vector<std::vector<std::shared_ptr<Element>>> batches;
  if (!GetTopologicalBatches(&batches).ok()) {
    return 0U;
  }
  std::size_t maximum = 0U;
  for (const auto &batch : batches) {
    maximum = std::max(maximum, batch.size());
  }
  return maximum;
}

std::size_t ElementManager::Trim() {
  std::vector<std::shared_ptr<Element>> elements;
  if (!GetElements(&elements).ok()) {
    return 0U;
  }
  std::size_t trim_count = 0U;
  for (const auto &element : elements) {
    std::vector<std::shared_ptr<Element>> dependencies;
    {
      std::lock_guard<std::mutex> lock(element->relation_mutex_);
      for (const auto &dependency : element->dependencies_) {
        const auto value = dependency.lock();
        if (value != nullptr) {
          dependencies.emplace_back(value);
        }
      }
    }
    for (const auto &dependency : dependencies) {
      if (!element->RemoveDependency(dependency).ok()) {
        continue;
      }
      if (dependency->HasPathTo(element)) {
        ++trim_count;
      } else {
        static_cast<void>(element->AddDependency(dependency));
      }
    }
  }
  return trim_count;
}

bool ElementManager::AreSeparated(const std::shared_ptr<Element> &first,
                                  const std::shared_ptr<Element> &second) const {
  OMNI_RETURN_VAL_IF(first == nullptr || second == nullptr || first == second, false);
  OMNI_RETURN_VAL_IF(!Find(first) || !Find(second), false);
  return first->HasPathTo(second) || second->HasPathTo(first);
}

bool ElementManager::IsSerializable() const {
  std::vector<std::shared_ptr<Element>> elements;
  OMNI_RETURN_VAL_IF(!GetElements(&elements).ok() || elements.empty(), false);
  std::size_t root_count = 0U;
  std::size_t tail_count = 0U;
  for (const auto &element : elements) {
    OMNI_RETURN_VAL_IF(!element->IsSerializable() || element->timeout_ms_ > kNoTimeoutMs, false);
    std::lock_guard<std::mutex> lock(element->relation_mutex_);
    OMNI_RETURN_VAL_IF(element->dependencies_.size() > 1U || element->successors_.size() > 1U, false);
    if (element->dependencies_.empty()) {
      ++root_count;
    }
    if (element->successors_.empty()) {
      ++tail_count;
    }
  }
  return root_count == 1U && tail_count == 1U;
}

Status ElementManager::Init() {
  OMNI_RETURN_VAL_IF_LOG(thread_pool_ == nullptr, InternalStatus("thread pool is unavailable"), ERROR,
                    "thread pool is unavailable");
  OMNI_RETURN_STATUS_IF_NOT_OK(Validate());
  std::vector<std::shared_ptr<Element>> elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(GetElements(&elements));
  std::stable_sort(
      elements.begin(), elements.end(),
      [](const std::shared_ptr<Element> &first, const std::shared_ptr<Element> &second) {
        return first->level() < second->level();
      });
  std::vector<std::shared_ptr<Element>> initialized;
  for (const auto &element : elements) {
    const Status status = element->Process(Element::Lifecycle::kInit);
    if (!status.ok()) {
      for (auto iter = initialized.rbegin(); iter != initialized.rend(); ++iter) {
        static_cast<void>((*iter)->Process(Element::Lifecycle::kDestroy));
      }
      return status;
    }
    initialized.emplace_back(element);
  }
  return Status();
}

Status ElementManager::Run() {
  OMNI_RETURN_VAL_IF_LOG(thread_pool_ == nullptr, InternalStatus("thread pool is unavailable"), ERROR,
                    "thread pool is unavailable");
  std::vector<std::vector<std::shared_ptr<Element>>> batches;
  OMNI_RETURN_STATUS_IF_NOT_OK(GetTopologicalBatches(&batches));
  const bool is_worker_thread = omni_runtime::utils::ThreadPool::IsWorkerThread();
  for (const auto &batch : batches) {
    if (!is_worker_thread && engine_type_ != EngineType::kTopological && batch.size() > 1U) {
      std::vector<std::future<Status>> futures;
      futures.reserve(batch.size());
      for (const auto &element : batch) {
        futures.emplace_back(thread_pool_->SafePost(
            [element]() { return element->Process(Element::Lifecycle::kRun); }));
      }
      Status first_error;
      for (auto &future : futures) {
        const Status status = future.get();
        if (first_error.ok() && !status.ok()) {
          first_error = status;
        }
      }
      OMNI_RETURN_STATUS_IF_NOT_OK(first_error);
    } else {
      for (const auto &element : batch) {
        OMNI_RETURN_STATUS_IF_NOT_OK(element->Process(Element::Lifecycle::kRun));
      }
    }
  }
  return Status();
}

Status ElementManager::Destroy() {
  OMNI_RETURN_STATUS_IF_NOT_OK(WaitAsyncElements());
  std::vector<std::shared_ptr<Element>> elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(GetElements(&elements));
  std::stable_sort(
      elements.begin(), elements.end(),
      [](const std::shared_ptr<Element> &first, const std::shared_ptr<Element> &second) {
        return first->level() > second->level();
      });
  Status first_error;
  for (const auto &element : elements) {
    const Status status = element->Process(Element::Lifecycle::kDestroy);
    if (first_error.ok() && !status.ok()) {
      first_error = status;
    }
  }
  return first_error;
}

Status ElementManager::GetTopologicalBatches(
    std::vector<std::vector<std::shared_ptr<Element>>> *const batches) const {
  OMNI_RETURN_VAL_IF_LOG(batches == nullptr, InvalidArgumentStatus("batches is null"), ERROR,
                    "batches is null");
  std::vector<std::shared_ptr<Element>> elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(GetElements(&elements));
  std::unordered_map<const Element *, std::size_t> index;
  for (std::size_t element_index = 0U; element_index < elements.size(); ++element_index) {
    OMNI_RETURN_VAL_IF_LOG(elements[element_index] == nullptr, InvalidArgumentStatus("element is null"),
                      ERROR, "element is null");
    index.emplace(elements[element_index].get(), element_index);
  }

  std::vector<std::size_t> indegree(elements.size(), 0U);
  std::vector<std::vector<std::size_t>> successors(elements.size());
  for (std::size_t element_index = 0U; element_index < elements.size(); ++element_index) {
    std::lock_guard<std::mutex> lock(elements[element_index]->relation_mutex_);
    for (const auto &dependency : elements[element_index]->dependencies_) {
      const auto dependency_element = dependency.lock();
      OMNI_RETURN_VAL_IF_LOG(dependency_element == nullptr, InvalidArgumentStatus("expired dependency"),
                        ERROR, "expired dependency");
      const auto iter = index.find(dependency_element.get());
      OMNI_RETURN_VAL_IF_LOG(iter == index.end(),
                        InvalidArgumentStatus("dependency is outside element manager"), ERROR,
                        "dependency is outside element manager");
      ++indegree[element_index];
      successors[iter->second].emplace_back(element_index);
    }
  }

  batches->clear();
  std::vector<std::size_t> ready;
  for (std::size_t element_index = 0U; element_index < indegree.size(); ++element_index) {
    if (indegree[element_index] == 0U) {
      ready.emplace_back(element_index);
    }
  }
  std::size_t processed_count = 0U;
  while (!ready.empty()) {
    std::vector<std::shared_ptr<Element>> batch;
    std::vector<std::size_t> next_ready;
    batch.reserve(ready.size());
    for (const std::size_t element_index : ready) {
      batch.emplace_back(elements[element_index]);
      ++processed_count;
      for (const std::size_t successor_index : successors[element_index]) {
        --indegree[successor_index];
        if (indegree[successor_index] == 0U) {
          next_ready.emplace_back(successor_index);
        }
      }
    }
    batches->emplace_back(std::move(batch));
    ready = std::move(next_ready);
  }
  OMNI_RETURN_VAL_IF_LOG(processed_count != elements.size(),
                    InvalidArgumentStatus("element graph contains a cycle"), ERROR,
                    "element graph contains a cycle");
  return Status();
}

} // namespace graph
} // namespace omni_runtime
