#include "Omni-Runtime/graph/pipeline/pipeline.h"

#include <iomanip>
#include <thread>

#include "Omni-Runtime/graph/pipeline/storage.h"

namespace omni_runtime {
namespace graph {
namespace {

Status AppendElementTree(const std::shared_ptr<Element> &element,
                         std::vector<std::shared_ptr<Element>> *const targets) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr || targets == nullptr,
                    InvalidArgumentStatus("aspect target input is null"), ERROR,
                    "aspect target input is null");
  targets->emplace_back(element);
  ElementRelation relation;
  OMNI_RETURN_STATUS_IF_NOT_OK(element->GetRelation(&relation));
  for (const auto &child : relation.children) {
    OMNI_RETURN_STATUS_IF_NOT_OK(AppendElementTree(child, targets));
  }
  return Status();
}

Status WriteElementPerf(const std::shared_ptr<Element> &element, std::ostream *const stream,
                        const std::size_t depth) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr || stream == nullptr,
                    InvalidArgumentStatus("perf output input is null"), ERROR,
                    "perf output input is null");
  const PerfInfo info = element->perf_info();
  *stream << std::string(depth * 2U, ' ') << element->name() << '\t' << info.run_count << '\t'
          << std::fixed << std::setprecision(3) << info.total_time_ms << '\t'
          << info.average_time_ms() << '\t' << info.minimum_time_ms << '\t' << info.maximum_time_ms
          << '\n';
  ElementRelation relation;
  OMNI_RETURN_STATUS_IF_NOT_OK(element->GetRelation(&relation));
  for (const auto &child : relation.children) {
    OMNI_RETURN_STATUS_IF_NOT_OK(WriteElementPerf(child, stream, depth + 1U));
  }
  return Status();
}

} // namespace

Pipeline::Pipeline()
    : Descriptor("pipeline"), element_manager_(std::make_shared<ElementManager>()),
      param_manager_(std::make_shared<ParamManager>()),
      daemon_manager_(std::make_shared<DaemonManager>()),
      event_manager_(std::make_shared<EventManager>()),
      stage_manager_(std::make_shared<StageManager>()) {
  const uint32_t hardware_threads = std::thread::hardware_concurrency();
  config_.thread_count = hardware_threads == 0U ? 1U : hardware_threads;
  static_cast<void>(stage_manager_->SetParamManager(param_manager_));
  static_cast<void>(daemon_manager_->SetContext(param_manager_, event_manager_));
}

Pipeline::~Pipeline() {
  if (is_initialized_) {
    static_cast<void>(Destroy());
  }
}

Status Pipeline::Init() {
  OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("pipeline is initialized"), ERROR,
                    "pipeline is initialized");
  OMNI_RETURN_VAL_IF_LOG(config_.thread_count == 0U, InvalidArgumentStatus("thread count is zero"),
                    ERROR, "thread count is zero");
  thread_pool_ = std::make_shared<omni_runtime::utils::ThreadPool>(name(), config_.thread_count);
  Status status;
  if (!event_manager_->SetContext(param_manager_, thread_pool_)) {
    status = InternalStatus("event context setup failed");
  } else if (!daemon_manager_->SetContext(param_manager_, event_manager_)) {
    status = InternalStatus("daemon context setup failed");
  } else if (!stage_manager_->SetParamManager(param_manager_)) {
    status = InternalStatus("stage context setup failed");
  } else if (!element_manager_->SetThreadPool(thread_pool_)) {
    status = InternalStatus("element thread pool setup failed");
  } else {
    status = ConfigureElements();
  }
  if (!status.ok()) {
    thread_pool_->Wait();
    thread_pool_->Stop();
    thread_pool_.reset();
    return status;
  }

  bool is_param_initialized = false;
  bool is_event_initialized = false;
  bool is_element_initialized = false;
  bool is_stage_initialized = false;
  status = param_manager_->Init();
  if (status.ok()) {
    is_param_initialized = true;
    status = event_manager_->Init();
  }
  if (status.ok()) {
    is_event_initialized = true;
    status = element_manager_->Init();
  }
  if (status.ok()) {
    is_element_initialized = true;
    status = stage_manager_->Init();
  }
  if (status.ok()) {
    is_stage_initialized = true;
    status = daemon_manager_->Init();
  }
  if (!status.ok()) {
    if (is_stage_initialized) {
      static_cast<void>(stage_manager_->Destroy());
    }
    if (is_element_initialized) {
      static_cast<void>(element_manager_->Destroy());
    }
    if (is_event_initialized) {
      static_cast<void>(event_manager_->Destroy());
    }
    if (is_param_initialized) {
      static_cast<void>(param_manager_->Destroy());
    }
    thread_pool_->Wait();
    thread_pool_->Stop();
    thread_pool_.reset();
    return status;
  }
  is_initialized_ = true;
  state_.store(PipelineState::kNormal, std::memory_order_release);
  return Status();
}

Status Pipeline::Run() {
  OMNI_RETURN_VAL_IF_LOG(!is_initialized_, InvalidArgumentStatus("pipeline is not initialized"), ERROR,
                    "pipeline is not initialized");
  OMNI_RETURN_STATUS_IF_NOT_OK(SetState(PipelineState::kNormal));
  OMNI_RETURN_STATUS_IF_NOT_OK(param_manager_->Setup());
  Status status = element_manager_->Run();
  const Status async_status = element_manager_->WaitAsyncElements();
  const Status event_status = event_manager_->Reset();
  if (status.ok() && !async_status.ok()) {
    status = async_status;
  }
  if (status.ok() && !event_status.ok()) {
    status = event_status;
  }
  OMNI_RETURN_VAL_IF_LOG(!param_manager_->Reset(status), InternalStatus("param reset failed"), ERROR,
                    "param reset failed");
  return status;
}

Status Pipeline::Destroy() {
  OMNI_RETURN_VAL_IF_LOG(!is_initialized_, InvalidArgumentStatus("pipeline is not initialized"), ERROR,
                    "pipeline is not initialized");
  Status first_error = daemon_manager_->Destroy();
  const Status element_status = element_manager_->Destroy();
  const Status event_status = event_manager_->Destroy();
  const Status stage_status = stage_manager_->Destroy();
  const Status param_status = param_manager_->Destroy();
  if (first_error.ok() && !element_status.ok()) {
    first_error = element_status;
  }
  if (first_error.ok() && !event_status.ok()) {
    first_error = event_status;
  }
  if (first_error.ok() && !stage_status.ok()) {
    first_error = stage_status;
  }
  if (first_error.ok() && !param_status.ok()) {
    first_error = param_status;
  }
  if (thread_pool_ != nullptr) {
    thread_pool_->Wait();
    thread_pool_->Stop();
  }
  thread_pool_.reset();
  is_initialized_ = false;
  state_.store(PipelineState::kNormal, std::memory_order_release);
  return first_error;
}

Status Pipeline::Process(const std::size_t run_count) {
  OMNI_RETURN_VAL_IF_LOG(run_count == 0U, InvalidArgumentStatus("run count is zero"), ERROR,
                    "run count is zero");
  OMNI_RETURN_STATUS_IF_NOT_OK(Init());
  Status status;
  for (std::size_t run_index = 0U; run_index < run_count; ++run_index) {
    if (state() == PipelineState::kCancel) {
      break;
    }
    status = Run();
    if (!status.ok()) {
      break;
    }
  }
  const Status destroy_status = Destroy();
  return status.ok() ? destroy_status : status;
}

std::future<Status> Pipeline::AsyncRun(const std::launch policy) {
  return std::async(policy, [this]() { return Run(); });
}

std::future<Status> Pipeline::AsyncProcess(const std::size_t run_count, const std::launch policy) {
  return std::async(policy, [this, run_count]() { return Process(run_count); });
}

bool Pipeline::set_thread_count(const uint32_t thread_count) {
  OMNI_RETURN_VAL_IF(is_initialized_ || thread_count == 0U, false);
  config_.thread_count = thread_count;
  return true;
}

bool Pipeline::set_engine_type(const EngineType engine_type) {
  OMNI_RETURN_VAL_IF(is_initialized_, false);
  return element_manager_->set_engine_type(engine_type);
}

Status Pipeline::Cancel() {
  OMNI_RETURN_VAL_IF_LOG(!is_initialized_, InvalidArgumentStatus("pipeline is not initialized"), ERROR,
                    "pipeline is not initialized");
  return SetState(PipelineState::kCancel);
}

Status Pipeline::Suspend() {
  OMNI_RETURN_VAL_IF_LOG(!is_initialized_, InvalidArgumentStatus("pipeline is not initialized"), ERROR,
                    "pipeline is not initialized");
  return SetState(PipelineState::kSuspend);
}

Status Pipeline::Resume() {
  OMNI_RETURN_VAL_IF_LOG(!is_initialized_, InvalidArgumentStatus("pipeline is not initialized"), ERROR,
                    "pipeline is not initialized");
  return SetState(PipelineState::kNormal);
}

Status Pipeline::Dump(std::ostream *const stream) const {
  OMNI_RETURN_VAL_IF_LOG(stream == nullptr, InvalidArgumentStatus("stream is null"), ERROR,
                    "stream is null");
  std::vector<std::shared_ptr<Element>> elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(element_manager_->GetElements(&elements));
  *stream << "digraph zomni_graph {\n";
  for (const auto &element : elements) {
    OMNI_RETURN_STATUS_IF_NOT_OK(element->Dump(stream));
  }
  *stream << "}\n";
  return Status();
}

Status Pipeline::Perf(std::ostream *const stream) const {
  OMNI_RETURN_VAL_IF_LOG(stream == nullptr, InvalidArgumentStatus("stream is null"), ERROR,
                    "stream is null");
  std::vector<std::shared_ptr<Element>> elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(element_manager_->GetElements(&elements));
  *stream << "element\truns\ttotal_ms\taverage_ms\tminimum_ms\tmaximum_ms\n";
  for (const auto &element : elements) {
    OMNI_RETURN_STATUS_IF_NOT_OK(WriteElementPerf(element, stream, 0U));
  }
  return Status();
}

Status Pipeline::Save(const std::string &path) const {
  return Storage::Save(this, path);
}

Status Pipeline::Load(const std::string &path) {
  return Storage::Load(path, this);
}

std::size_t Pipeline::MaxParallelism() const {
  OMNI_RETURN_VAL_IF(is_initialized_, 0U);
  return element_manager_->MaxParallelism();
}

std::size_t Pipeline::Trim() {
  OMNI_RETURN_VAL_IF(is_initialized_, 0U);
  return element_manager_->Trim();
}

Status Pipeline::MakeSerial() {
  OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("pipeline is initialized"), ERROR,
                    "pipeline is initialized");
  OMNI_RETURN_VAL_IF_LOG(!element_manager_->IsSerializable(),
                    InvalidArgumentStatus("pipeline is not serializable"), ERROR,
                    "pipeline is not serializable");
  config_.thread_count = 1U;
  element_manager_->set_engine_type(EngineType::kTopological);
  return Status();
}

bool Pipeline::AreSeparated(const std::shared_ptr<Element> &first,
                            const std::shared_ptr<Element> &second) const {
  return element_manager_->AreSeparated(first, second);
}

Status Pipeline::RegisterElement(const std::shared_ptr<Element> &element,
                                 const std::vector<std::shared_ptr<Element>> &dependencies,
                                 const std::string &name, const std::size_t loop_count) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr, InvalidArgumentStatus("element is null"), ERROR,
                    "element is null");
  OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("pipeline is initialized"), ERROR,
                    "pipeline is initialized");
  OMNI_RETURN_VAL_IF_LOG(element->IsRegistered(), InvalidArgumentStatus("element is registered"), ERROR,
                    "element is registered");
  OMNI_RETURN_VAL_IF_LOG(!element->set_name(name) || !element->set_loop_count(loop_count),
                    InvalidArgumentStatus("element configuration failed"), ERROR,
                    "element configuration failed");
  for (const auto &dependency : dependencies) {
    OMNI_RETURN_STATUS_IF_NOT_OK(element->AddDependency(dependency));
  }
  return element_manager_->Add(element);
}

Status Pipeline::SetState(const PipelineState state) {
  OMNI_RETURN_STATUS_IF_NOT_OK(element_manager_->SetState(state));
  state_.store(state, std::memory_order_release);
  return Status();
}

Status Pipeline::ConfigureElements() {
  std::vector<std::shared_ptr<Element>> elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(element_manager_->GetElements(&elements));
  for (const auto &element : elements) {
    OMNI_RETURN_VAL_IF_LOG(
        !element->SetContext(param_manager_, event_manager_, stage_manager_, thread_pool_),
        InternalStatus("element context setup failed"), ERROR,
        "element context setup failed: " << element->name());
  }
  return Status();
}

Status
Pipeline::GetGlobalAspectTargets(std::vector<std::shared_ptr<Element>> *const targets) const {
  OMNI_RETURN_VAL_IF_LOG(targets == nullptr, InvalidArgumentStatus("aspect targets are null"), ERROR,
                    "aspect targets are null");
  std::vector<std::shared_ptr<Element>> elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(element_manager_->GetElements(&elements));
  targets->clear();
  for (const auto &element : elements) {
    OMNI_RETURN_STATUS_IF_NOT_OK(AppendElementTree(element, targets));
  }
  return Status();
}

} // namespace graph
} // namespace omni_runtime
