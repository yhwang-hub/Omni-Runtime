#include "Omni-Runtime/graph/element/element.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <unordered_set>

#include "Omni-Runtime/utils/status_macros.h"

namespace omni_runtime {
namespace graph {

Element::Element(const ElementType element_type)
    : Descriptor("element"), element_type_(element_type) {
}

Element::~Element() {
  static_cast<void>(WaitAsync());
}

bool Element::set_name(const std::string &name) {
  OMNI_RETURN_VAL_IF(is_initialized_, false);
  return Descriptor::set_name(name);
}

bool Element::set_loop_count(const std::size_t loop_count) {
  OMNI_RETURN_VAL_IF(is_initialized_ || loop_count == 0U ||
                    (timeout_ms_ > kNoTimeoutMs && loop_count != kDefaultLoopCount),
                false);
  loop_count_ = loop_count;
  return true;
}

bool Element::set_level(const Level level) {
  OMNI_RETURN_VAL_IF(is_initialized_, false);
  level_ = level;
  return true;
}

bool Element::set_visible(const bool is_visible) {
  is_visible_ = is_visible;
  return true;
}

bool Element::set_timeout(const Milliseconds timeout_ms,
                          const ElementTimeoutStrategy timeout_strategy) {
  OMNI_RETURN_VAL_IF(is_initialized_ || timeout_ms < 0 ||
                    (loop_count_ > kDefaultLoopCount && timeout_ms != kNoTimeoutMs),
                false);
  timeout_ms_ = timeout_ms;
  timeout_strategy_ = timeout_strategy;
  return true;
}

bool Element::set_binding_index(const Index binding_index) {
  OMNI_RETURN_VAL_IF(is_initialized_, false);
  binding_index_ = binding_index;
  return true;
}

bool Element::set_macro(const bool is_macro) {
  OMNI_RETURN_VAL_IF(is_initialized_ || IsGroup(), false);
  is_macro_ = is_macro;
  return true;
}

ElementState Element::state() const {
  return IsTimeout() ? ElementState::kTimeout : state_.load(std::memory_order_acquire);
}

PerfInfo Element::perf_info() const {
  std::lock_guard<std::mutex> lock(perf_mutex_);
  PerfInfo result = perf_info_;
  if (result.run_count == 0U) {
    result.minimum_time_ms = 0.0;
  }
  return result;
}

bool Element::IsGroup() const {
  return (static_cast<uint32_t>(element_type_) & static_cast<uint32_t>(ElementType::kGroup)) != 0U;
}

bool Element::IsAdapter() const {
  return (static_cast<uint32_t>(element_type_) & static_cast<uint32_t>(ElementType::kAdapter)) !=
         0U;
}

bool Element::IsNode() const {
  return element_type_ == ElementType::kNode;
}

bool Element::IsTimeout() const {
  if (state_.load(std::memory_order_acquire) == ElementState::kTimeout) {
    return true;
  }
  auto belong = belong_.lock();
  while (belong != nullptr) {
    if (belong->state_.load(std::memory_order_acquire) == ElementState::kTimeout) {
      return true;
    }
    belong = belong->belong_.lock();
  }
  return false;
}

Status Element::AddDependency(const std::shared_ptr<Element> &element) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr, InvalidArgumentStatus("dependency is null"), ERROR,
                    "dependency is null");
  const auto owner = belong_.lock();
  const bool is_mutable = owner != nullptr && owner->element_type() == ElementType::kMutable;
  OMNI_RETURN_VAL_IF_LOG(is_initialized_ && !is_mutable, InvalidArgumentStatus("element is initialized"),
                    ERROR, "element is initialized");
  if (element.get() == this) {
    return Status();
  }
  const auto belong = belong_.lock();
  const auto dependency_belong = element->belong_.lock();
  OMNI_RETURN_VAL_IF_LOG(belong != dependency_belong,
                    InvalidArgumentStatus("elements have different owners"), ERROR,
                    "elements have different owners");
  OMNI_RETURN_VAL_IF_LOG(HasPathTo(element), InvalidArgumentStatus("dependency creates a cycle"), ERROR,
                    "dependency creates a cycle");
  {
    std::lock_guard<std::mutex> lock(relation_mutex_);
    const auto iter = std::find_if(dependencies_.begin(), dependencies_.end(),
                                   [&element](const std::weak_ptr<Element> &dependency) {
                                     return dependency.lock() == element;
                                   });
    if (iter == dependencies_.end()) {
      dependencies_.emplace_back(element);
    }
  }
  OMNI_RETURN_VAL_IF_LOG(!element->AddSuccessor(shared_from_this()),
                    InternalStatus("failed to add successor"), ERROR, "failed to add successor");
  return Status();
}

Status Element::RemoveDependency(const std::shared_ptr<Element> &element) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr, InvalidArgumentStatus("dependency is null"), ERROR,
                    "dependency is null");
  const auto owner = belong_.lock();
  const bool is_mutable = owner != nullptr && owner->element_type() == ElementType::kMutable;
  OMNI_RETURN_VAL_IF_LOG(is_initialized_ && !is_mutable, InvalidArgumentStatus("element is initialized"),
                    ERROR, "element is initialized");
  bool is_found = false;
  {
    std::lock_guard<std::mutex> lock(relation_mutex_);
    const auto iter = std::remove_if(dependencies_.begin(), dependencies_.end(),
                                     [&element](const std::weak_ptr<Element> &dependency) {
                                       return dependency.lock() == element;
                                     });
    is_found = iter != dependencies_.end();
    dependencies_.erase(iter, dependencies_.end());
  }
  OMNI_RETURN_VAL_IF_LOG(!is_found, NotFoundStatus("dependency not found"), ERROR,
                    "dependency not found");
  OMNI_RETURN_VAL_IF_LOG(!element->RemoveSuccessor(shared_from_this()),
                    InternalStatus("failed to remove successor"), ERROR,
                    "failed to remove successor");
  return Status();
}

Status Element::GetRelation(ElementRelation *const relation) const {
  OMNI_RETURN_VAL_IF_LOG(relation == nullptr, InvalidArgumentStatus("relation is null"), ERROR,
                    "relation is null");
  relation->predecessors.clear();
  relation->successors.clear();
  {
    std::lock_guard<std::mutex> lock(relation_mutex_);
    for (const auto &dependency : dependencies_) {
      const auto value = dependency.lock();
      if (value != nullptr) {
        relation->predecessors.emplace_back(value);
      }
    }
    for (const auto &successor : successors_) {
      const auto value = successor.lock();
      if (value != nullptr) {
        relation->successors.emplace_back(value);
      }
    }
  }
  relation->children = ChildElements();
  relation->belong = belong_.lock();
  return Status();
}

Status Element::AddAspect(const std::shared_ptr<Aspect> &aspect, const AspectParam *const param) {
  OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("element is initialized"), ERROR,
                    "element is initialized");
  if (aspect_manager_ == nullptr) {
    aspect_manager_ = std::make_shared<AspectManager>();
    if (IsRegistered()) {
      OMNI_RETURN_VAL_IF_LOG(!aspect_manager_->SetContext(param_manager_, event_manager_, name()),
                        InternalStatus("aspect context setup failed"), ERROR,
                        "aspect context setup failed");
    }
  }
  return aspect_manager_->Add(aspect, param);
}

Status Element::PopLastAspect() {
  OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("element is initialized"), ERROR,
                    "element is initialized");
  OMNI_RETURN_VAL_IF_LOG(aspect_manager_ == nullptr, NotFoundStatus("aspect manager is empty"), ERROR,
                    "aspect manager is empty");
  OMNI_RETURN_STATUS_IF_NOT_OK(aspect_manager_->PopLast());
  if (aspect_manager_->size() == 0U) {
    aspect_manager_.reset();
  }
  return Status();
}

Status Element::Notify(const std::string &key, const EventType type,
                       const EventAsyncStrategy strategy) const {
  OMNI_RETURN_VAL_IF_LOG(event_manager_ == nullptr, InternalStatus("event manager is unavailable"),
                    ERROR, "event manager is unavailable");
  return event_manager_->Trigger(key, type, strategy);
}

Status Element::EnterStage(const std::string &key) const {
  OMNI_RETURN_VAL_IF_LOG(stage_manager_ == nullptr, InternalStatus("stage manager is unavailable"),
                    ERROR, "stage manager is unavailable");
  return stage_manager_->WaitForReady(key);
}

Status Element::Spawn(const TaskGroup &tasks, const Milliseconds timeout_ms) const {
  OMNI_RETURN_VAL_IF_LOG(thread_pool_ == nullptr, InternalStatus("thread pool is unavailable"), ERROR,
                    "thread pool is unavailable");
  OMNI_RETURN_VAL_IF_LOG(timeout_ms < 0, InvalidArgumentStatus("timeout is negative"), ERROR,
                    "timeout is negative");
  std::vector<std::future<Status>> futures;
  futures.reserve(tasks.size());
  for (const auto &task : tasks) {
    OMNI_RETURN_VAL_IF_LOG(!task, InvalidArgumentStatus("task is empty"), ERROR, "task is empty");
    futures.emplace_back(thread_pool_->SafePost(task));
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  for (auto &future : futures) {
    OMNI_RETURN_VAL_IF_LOG(future.wait_until(deadline) != std::future_status::ready,
                      Status::Cancelled("task group timeout", SOURCE_LOCATION()), ERROR,
                      "task group timeout");
    OMNI_RETURN_STATUS_IF_NOT_OK(future.get());
  }
  return Status();
}

Status Element::Process(const Lifecycle lifecycle) {
  if (!is_visible_) {
    return Status();
  }
  try {
    if (lifecycle == Lifecycle::kInit) {
      OMNI_RETURN_STATUS_IF_NOT_OK(Reflect(AspectType::kBeginInit, Status()));
      const Status status = Init();
      const Status aspect_status = Reflect(AspectType::kFinishInit, status);
      OMNI_RETURN_STATUS_IF_NOT_OK(status);
      OMNI_RETURN_STATUS_IF_NOT_OK(aspect_status);
      is_initialized_ = true;
      is_prepared_ = false;
      return Status();
    }
    if (lifecycle == Lifecycle::kRun) {
      return ProcessRun();
    }
    OMNI_RETURN_STATUS_IF_NOT_OK(Reflect(AspectType::kBeginDestroy, Status()));
    const Status status = Destroy();
    const Status aspect_status = Reflect(AspectType::kFinishDestroy, status);
    OMNI_RETURN_STATUS_IF_NOT_OK(status);
    OMNI_RETURN_STATUS_IF_NOT_OK(aspect_status);
    is_initialized_ = false;
    return Status();
  } catch (const std::exception &exception) {
    static_cast<void>(Reflect(AspectType::kEnterCrashed, Status()));
    return InternalStatus(exception.what());
  } catch (...) {
    static_cast<void>(Reflect(AspectType::kEnterCrashed, Status()));
    return InternalStatus("unknown element exception");
  }
}

Status Element::ProcessRun() {
  OMNI_RETURN_VAL_IF_LOG(!is_initialized_, InvalidArgumentStatus("element is not initialized"), ERROR,
                    "element is not initialized: " << name());
  if (!is_prepared_) {
    OMNI_RETURN_STATUS_IF_NOT_OK(PrepareRun());
    is_prepared_ = true;
  }
  for (std::size_t loop_index = 0U; loop_index < loop_count_; ++loop_index) {
    OMNI_RETURN_VAL_IF_LOG(state() == ElementState::kCancel,
                      Status::Cancelled("element is cancelled", SOURCE_LOCATION()), WARN,
                      "element is cancelled: " << name());
    OMNI_RETURN_VAL_IF_LOG(!WaitIfSuspended(), InternalStatus("suspend wait failed"), ERROR,
                      "suspend wait failed");
    OMNI_RETURN_STATUS_IF_NOT_OK(Reflect(AspectType::kBeginRun, Status()));
    const auto start_time = std::chrono::steady_clock::now();
    Status status;
    do {
      status = timeout_ms_ > kNoTimeoutMs ? RunWithTimeout() : Run();
      OMNI_RETURN_VAL_IF_LOG(!WaitIfSuspended(), InternalStatus("suspend wait failed"), ERROR,
                        "suspend wait failed");
    } while (status.ok() && IsHold());
    const auto finish_time = std::chrono::steady_clock::now();
    const double elapsed_time_ms =
        std::chrono::duration<double, std::milli>(finish_time - start_time).count();
    static_cast<void>(RecordRunTime(elapsed_time_ms));
    const Status aspect_status = Reflect(AspectType::kFinishRun, status);
    OMNI_RETURN_STATUS_IF_NOT_OK(status);
    OMNI_RETURN_STATUS_IF_NOT_OK(aspect_status);
  }
  OMNI_RETURN_STATUS_IF_NOT_OK(CheckRunResult());
  is_done_.store(true, std::memory_order_release);
  return Status();
}

Status Element::RunWithTimeout() {
  OMNI_RETURN_VAL_IF_LOG(thread_pool_ == nullptr, InternalStatus("thread pool is unavailable"), ERROR,
                    "thread pool is unavailable");
  {
    std::lock_guard<std::mutex> lock(async_mutex_);
    async_result_ = thread_pool_->SafePost([this]() { return Run(); }).share();
  }
  if (async_result_.wait_for(std::chrono::milliseconds(timeout_ms_)) == std::future_status::ready) {
    return WaitAsync();
  }
  OMNI_RETURN_STATUS_IF_NOT_OK(Reflect(AspectType::kEnterTimeout, Status()));
  state_.store(ElementState::kTimeout, std::memory_order_release);
  if (timeout_strategy_ == ElementTimeoutStrategy::kAsError) {
    return Status::Cancelled("element timeout: " + name(), SOURCE_LOCATION());
  }
  return Status();
}

Status Element::WaitAsync() {
  std::shared_future<Status> future;
  {
    std::lock_guard<std::mutex> lock(async_mutex_);
    future = async_result_;
  }
  if (!future.valid()) {
    return Status();
  }
  const Status status = future.get();
  {
    std::lock_guard<std::mutex> lock(async_mutex_);
    async_result_ = std::shared_future<Status>();
  }
  return status;
}

Status Element::Reflect(const AspectType type, const Status &status) const {
  if (aspect_manager_ == nullptr || aspect_manager_->size() == 0U) {
    return Status();
  }
  return aspect_manager_->Reflect(type, status);
}

bool Element::SetContext(const std::shared_ptr<ParamManager> &param_manager,
                         const std::shared_ptr<EventManager> &event_manager,
                         const std::shared_ptr<StageManager> &stage_manager,
                         const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool) {
  OMNI_RETURN_VAL_IF(param_manager == nullptr || event_manager == nullptr || stage_manager == nullptr ||
                    thread_pool == nullptr,
                false);
  param_manager_ = param_manager;
  event_manager_ = event_manager;
  stage_manager_ = stage_manager;
  thread_pool_ = thread_pool;
  if (aspect_manager_ != nullptr) {
    OMNI_RETURN_VAL_IF(!aspect_manager_->SetContext(param_manager, event_manager, name()), false);
  }
  return SetContextExtra(thread_pool);
}

bool Element::SetState(const ElementState state) {
  state_.store(state, std::memory_order_release);
  if (state != ElementState::kSuspend) {
    suspend_condition_.notify_all();
  }
  return true;
}

bool Element::WaitIfSuspended() {
  if (state_.load(std::memory_order_acquire) != ElementState::kSuspend) {
    return true;
  }
  std::unique_lock<std::mutex> lock(suspend_mutex_);
  suspend_condition_.wait(
      lock, [this]() { return state_.load(std::memory_order_acquire) != ElementState::kSuspend; });
  return true;
}

bool Element::AddSuccessor(const std::shared_ptr<Element> &element) {
  OMNI_RETURN_VAL_IF(element == nullptr, false);
  std::lock_guard<std::mutex> lock(relation_mutex_);
  const auto iter = std::find_if(
      successors_.begin(), successors_.end(),
      [&element](const std::weak_ptr<Element> &successor) { return successor.lock() == element; });
  if (iter == successors_.end()) {
    successors_.emplace_back(element);
  }
  return true;
}

bool Element::RemoveSuccessor(const std::shared_ptr<Element> &element) {
  OMNI_RETURN_VAL_IF(element == nullptr, false);
  std::lock_guard<std::mutex> lock(relation_mutex_);
  const auto iter = std::remove_if(
      successors_.begin(), successors_.end(),
      [&element](const std::weak_ptr<Element> &successor) { return successor.lock() == element; });
  successors_.erase(iter, successors_.end());
  return true;
}

bool Element::SetBelong(const std::shared_ptr<Element> &belong) {
  belong_ = belong;
  return true;
}

bool Element::ClearRelations() {
  std::vector<std::shared_ptr<Element>> dependencies;
  {
    std::lock_guard<std::mutex> lock(relation_mutex_);
    for (const auto &dependency : dependencies_) {
      const auto value = dependency.lock();
      if (value != nullptr) {
        dependencies.emplace_back(value);
      }
    }
    dependencies_.clear();
    successors_.clear();
  }
  for (const auto &dependency : dependencies) {
    dependency->RemoveSuccessor(shared_from_this());
  }
  return true;
}

bool Element::IsRegistered() const {
  return param_manager_ != nullptr && event_manager_ != nullptr && stage_manager_ != nullptr;
}

bool Element::HasPathTo(const std::shared_ptr<Element> &target) const {
  OMNI_RETURN_VAL_IF(target == nullptr, false);
  std::vector<std::shared_ptr<Element>> pending;
  std::unordered_set<const Element *> visited;
  pending.emplace_back(const_cast<Element *>(this)->shared_from_this());
  while (!pending.empty()) {
    const auto current = pending.back();
    pending.pop_back();
    if (current == target) {
      return true;
    }
    if (!visited.emplace(current.get()).second) {
      continue;
    }
    std::lock_guard<std::mutex> lock(current->relation_mutex_);
    for (const auto &successor : current->successors_) {
      const auto value = successor.lock();
      if (value != nullptr) {
        pending.emplace_back(value);
      }
    }
  }
  return false;
}

bool Element::RecordRunTime(const double elapsed_time_ms) {
  OMNI_RETURN_VAL_IF(elapsed_time_ms < 0.0, false);
  std::lock_guard<std::mutex> lock(perf_mutex_);
  ++perf_info_.run_count;
  perf_info_.total_time_ms += elapsed_time_ms;
  perf_info_.minimum_time_ms = std::min(perf_info_.minimum_time_ms, elapsed_time_ms);
  perf_info_.maximum_time_ms = std::max(perf_info_.maximum_time_ms, elapsed_time_ms);
  return true;
}

Status Element::Dump(std::ostream *const stream) const {
  OMNI_RETURN_VAL_IF_LOG(stream == nullptr, InvalidArgumentStatus("stream is null"), ERROR,
                    "stream is null");
  *stream << 'p' << this << " [label=\"" << name() << "\"]\n";
  std::lock_guard<std::mutex> lock(relation_mutex_);
  for (const auto &successor : successors_) {
    const auto value = successor.lock();
    if (value != nullptr) {
      *stream << 'p' << this << " -> p" << value.get() << "\n";
    }
  }
  return Status();
}

} // namespace graph
} // namespace omni_runtime
