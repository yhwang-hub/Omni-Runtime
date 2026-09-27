#pragma once

#include <deque>
#include <memory>
#include <mutex>

#include "Omni-Runtime/graph/pipeline/pipeline.h"

namespace omni_runtime {
namespace graph {

class PipelineManager final : public Object {
public:
  PipelineManager() = default;
  ~PipelineManager() override;

  Status Add(const std::shared_ptr<Pipeline> &pipeline);
  Status Remove(const std::shared_ptr<Pipeline> &pipeline);
  bool Find(const std::shared_ptr<Pipeline> &pipeline) const;
  std::size_t size() const;
  std::shared_ptr<Pipeline> Fetch();
  Status Release(const std::shared_ptr<Pipeline> &pipeline);
  Status Clear();

  Status Init() final;
  Status Run() final;
  Status Destroy() final;

private:
  bool FindUnlocked(const std::shared_ptr<Pipeline> &pipeline) const;

  mutable std::mutex mutex_;
  std::deque<std::shared_ptr<Pipeline>> free_pipelines_;
  std::deque<std::shared_ptr<Pipeline>> used_pipelines_;
  bool is_initialized_ = false;
};

} // namespace graph
} // namespace omni_runtime
