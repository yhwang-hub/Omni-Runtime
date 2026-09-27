#include "Omni-Runtime/graph/pipeline/pipeline_factory.h"

#include <algorithm>

namespace omni_runtime {
namespace graph {

std::mutex PipelineFactory::mutex_;
std::vector<std::shared_ptr<Pipeline>> PipelineFactory::pipelines_;

std::shared_ptr<Pipeline> PipelineFactory::Create() {
  auto pipeline = std::make_shared<Pipeline>();
  std::lock_guard<std::mutex> lock(mutex_);
  pipelines_.emplace_back(pipeline);
  return pipeline;
}

Status PipelineFactory::Remove(const std::shared_ptr<Pipeline> &pipeline) {
  OMNI_RETURN_VAL_IF_LOG(pipeline == nullptr, InvalidArgumentStatus("pipeline is null"), ERROR,
                    "pipeline is null");
  std::lock_guard<std::mutex> lock(mutex_);
  const auto iter = std::find(pipelines_.begin(), pipelines_.end(), pipeline);
  OMNI_RETURN_VAL_IF_LOG(iter == pipelines_.end(), NotFoundStatus("pipeline not found"), ERROR,
                    "pipeline not found");
  pipelines_.erase(iter);
  return Status();
}

Status PipelineFactory::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  pipelines_.clear();
  return Status();
}

} // namespace graph
} // namespace omni_runtime
