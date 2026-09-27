#include "Omni-Runtime/graph/pipeline/pipeline_manager.h"

#include <algorithm>

#include "Omni-Runtime/utils/status_macros.h"

namespace omni_runtime {
namespace graph {

PipelineManager::~PipelineManager() {
  if (is_initialized_) {
    static_cast<void>(Destroy());
  }
  static_cast<void>(Clear());
}

Status PipelineManager::Add(const std::shared_ptr<Pipeline> &pipeline) {
  OMNI_RETURN_VAL_IF_LOG(pipeline == nullptr, InvalidArgumentStatus("pipeline is null"), ERROR,
                    "pipeline is null");
  OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("manager is initialized"), ERROR,
                    "manager is initialized");
  std::lock_guard<std::mutex> lock(mutex_);
  OMNI_RETURN_VAL_IF_LOG(FindUnlocked(pipeline), InvalidArgumentStatus("pipeline is duplicated"), ERROR,
                    "pipeline is duplicated");
  free_pipelines_.emplace_back(pipeline);
  return Status();
}

Status PipelineManager::Remove(const std::shared_ptr<Pipeline> &pipeline) {
  OMNI_RETURN_VAL_IF_LOG(pipeline == nullptr, InvalidArgumentStatus("pipeline is null"), ERROR,
                    "pipeline is null");
  OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("manager is initialized"), ERROR,
                    "manager is initialized");
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = std::find(free_pipelines_.begin(), free_pipelines_.end(), pipeline);
  OMNI_RETURN_VAL_IF_LOG(iter == free_pipelines_.end(), NotFoundStatus("pipeline not found"), ERROR,
                    "pipeline not found");
  free_pipelines_.erase(iter);
  return Status();
}

bool PipelineManager::Find(const std::shared_ptr<Pipeline> &pipeline) const {
  OMNI_RETURN_VAL_IF(pipeline == nullptr, false);
  std::lock_guard<std::mutex> lock(mutex_);
  return FindUnlocked(pipeline);
}

bool PipelineManager::FindUnlocked(const std::shared_ptr<Pipeline> &pipeline) const {
  const bool is_free =
      std::find(free_pipelines_.begin(), free_pipelines_.end(), pipeline) != free_pipelines_.end();
  const bool is_used =
      std::find(used_pipelines_.begin(), used_pipelines_.end(), pipeline) != used_pipelines_.end();
  return is_free || is_used;
}

std::size_t PipelineManager::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return free_pipelines_.size() + used_pipelines_.size();
}

std::shared_ptr<Pipeline> PipelineManager::Fetch() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (free_pipelines_.empty()) {
    return nullptr;
  }
  auto pipeline = free_pipelines_.front();
  free_pipelines_.pop_front();
  used_pipelines_.emplace_back(pipeline);
  return pipeline;
}

Status PipelineManager::Release(const std::shared_ptr<Pipeline> &pipeline) {
  OMNI_RETURN_VAL_IF_LOG(pipeline == nullptr, InvalidArgumentStatus("pipeline is null"), ERROR,
                    "pipeline is null");
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = std::find(used_pipelines_.begin(), used_pipelines_.end(), pipeline);
  OMNI_RETURN_VAL_IF_LOG(iter == used_pipelines_.end(), NotFoundStatus("used pipeline not found"), ERROR,
                    "used pipeline not found");
  used_pipelines_.erase(iter);
  free_pipelines_.emplace_front(pipeline);
  return Status();
}

Status PipelineManager::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  OMNI_RETURN_VAL_IF_LOG(!used_pipelines_.empty(), InvalidArgumentStatus("pipelines are in use"), ERROR,
                    "pipelines are in use");
  free_pipelines_.clear();
  return Status();
}

Status PipelineManager::Init() {
  OMNI_RETURN_VAL_IF_LOG(is_initialized_, InvalidArgumentStatus("manager is initialized"), ERROR,
                    "manager is initialized");
  std::vector<std::shared_ptr<Pipeline>> pipelines;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    OMNI_RETURN_VAL_IF_LOG(!used_pipelines_.empty(), InvalidArgumentStatus("pipelines are in use"),
                      ERROR, "pipelines are in use");
    pipelines.assign(free_pipelines_.begin(), free_pipelines_.end());
  }
  std::vector<std::shared_ptr<Pipeline>> initialized_pipelines;
  for (const auto &pipeline : pipelines) {
    const Status status = pipeline->Init();
    if (!status.ok()) {
      for (auto iter = initialized_pipelines.rbegin(); iter != initialized_pipelines.rend();
           ++iter) {
        static_cast<void>((*iter)->Destroy());
      }
      return status;
    }
    initialized_pipelines.emplace_back(pipeline);
  }
  is_initialized_ = true;
  return Status();
}

Status PipelineManager::Run() {
  OMNI_RETURN_VAL_IF_LOG(!is_initialized_, InvalidArgumentStatus("manager is not initialized"), ERROR,
                    "manager is not initialized");
  const auto pipeline = Fetch();
  OMNI_RETURN_VAL_IF_LOG(pipeline == nullptr, NotFoundStatus("free pipeline not found"), ERROR,
                    "free pipeline not found");
  const Status status = pipeline->Run();
  const Status release_status = Release(pipeline);
  return status.ok() ? release_status : status;
}

Status PipelineManager::Destroy() {
  OMNI_RETURN_VAL_IF_LOG(!is_initialized_, InvalidArgumentStatus("manager is not initialized"), ERROR,
                    "manager is not initialized");
  std::vector<std::shared_ptr<Pipeline>> pipelines;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    OMNI_RETURN_VAL_IF_LOG(!used_pipelines_.empty(), InvalidArgumentStatus("pipelines are in use"),
                      ERROR, "pipelines are in use");
    pipelines.assign(free_pipelines_.begin(), free_pipelines_.end());
  }
  Status first_error;
  for (auto iter = pipelines.rbegin(); iter != pipelines.rend(); ++iter) {
    const Status status = (*iter)->Destroy();
    if (first_error.ok() && !status.ok()) {
      first_error = status;
    }
  }
  is_initialized_ = false;
  return first_error;
}

} // namespace graph
} // namespace omni_runtime
