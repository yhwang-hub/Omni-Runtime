#include "Omni-Runtime/inference/inference_graph/inference_graph.h"

#include "Omni-Runtime/utils/utils.h"
#include "Omni-Runtime/graph/pipeline/pipeline_factory.h"

namespace omni_runtime {
namespace inference {

InferenceGraph::~InferenceGraph() {
  if (pipeline_ != nullptr) {
    static_cast<void>(graph::PipelineFactory::Remove(pipeline_));
  }
}

bool InferenceGraph::Init(const std::string &name, const uint32_t thread_count) {
  error_message_.clear();
  OMNI_RETURN_VAL_IF(name.empty() || thread_count == 0U,
                SetError(name.empty() ? "inference graph name is empty"
                                      : "inference graph thread count is zero"));
  if (pipeline_ != nullptr) {
    static_cast<void>(graph::PipelineFactory::Remove(pipeline_));
    pipeline_.reset();
  }
  pipeline_ = graph::PipelineFactory::Create();
  OMNI_RETURN_VAL_IF(pipeline_ == nullptr, SetError("failed to create inference graph pipeline"));
  OMNI_RETURN_VAL_IF(!pipeline_->set_name(name) || !pipeline_->set_thread_count(thread_count),
                SetError("failed to configure inference graph pipeline"));
  return true;
}

bool InferenceGraph::Process() {
  OMNI_RETURN_VAL_IF(pipeline_ == nullptr, SetError("inference graph is not initialized"));
  const graph::Status status = pipeline_->Process(1U);
  OMNI_RETURN_VAL_IF(!status.ok(), SetError(status.message()));
  return true;
}

bool InferenceGraph::SetError(const std::string &message) {
  error_message_ = message;
  return false;
}

const std::string &InferenceGraph::error_message() const {
  return error_message_;
}

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
