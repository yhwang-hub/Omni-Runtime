#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "Omni-Runtime/graph/pipeline/pipeline.h"

namespace omni_runtime {
namespace graph {

class PipelineFactory final {
public:
  static std::shared_ptr<Pipeline> Create();
  static Status Remove(const std::shared_ptr<Pipeline> &pipeline);
  static Status Clear();

private:
  static std::mutex mutex_;
  static std::vector<std::shared_ptr<Pipeline>> pipelines_;
};

} // namespace graph
} // namespace omni_runtime
