#pragma once

#include <memory>
#include <string>
#include <utility>

#include "Omni-Runtime/graph/element/node.h"
#include "Omni-Runtime/graph/pipeline/pipeline.h"

namespace omni_runtime {
namespace inference {

class InferenceGraph final {
public:
  InferenceGraph() = default;
  ~InferenceGraph();
  InferenceGraph(const InferenceGraph &) = delete;
  InferenceGraph &operator=(const InferenceGraph &) = delete;

  bool Init(const std::string &name, const uint32_t thread_count);

  template <typename TNode, typename... Args>
  std::shared_ptr<TNode> RegisterNode(const graph::NodeInfo &info, Args &&...args) {
    OMNI_RETURN_VAL_IF(pipeline_ == nullptr, nullptr);
    return pipeline_->RegisterNode<TNode>(info, std::forward<Args>(args)...);
  }

  bool Process();
  const std::string &error_message() const;

private:
  bool SetError(const std::string &message);

  std::shared_ptr<graph::Pipeline> pipeline_;
  std::string error_message_;
};

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
