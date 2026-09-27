#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/graph/element/element.h"

namespace omni_runtime {
namespace graph {

class ElementManager final : public Object {
public:
  ElementManager() = default;
  ~ElementManager() override;

  Status Add(const std::shared_ptr<Element> &element);
  Status Remove(const std::shared_ptr<Element> &element);
  bool Find(const std::shared_ptr<Element> &element) const;
  Status Clear();
  bool set_engine_type(const EngineType engine_type);
  EngineType engine_type() const {
    return engine_type_;
  }
  bool SetThreadPool(const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool);
  Status GetElements(std::vector<std::shared_ptr<Element>> *const elements) const;
  Status SetState(const ElementState state);
  Status WaitAsyncElements();
  Status Validate() const;
  std::size_t MaxParallelism() const;
  std::size_t Trim();
  bool AreSeparated(const std::shared_ptr<Element> &first,
                    const std::shared_ptr<Element> &second) const;
  bool IsSerializable() const;

  Status Init() final;
  Status Run() final;
  Status Destroy() final;

private:
  Status
  GetTopologicalBatches(std::vector<std::vector<std::shared_ptr<Element>>> *const batches) const;

  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<Element>> elements_;
  std::shared_ptr<omni_runtime::utils::ThreadPool> thread_pool_;
  EngineType engine_type_ = EngineType::kDynamic;
};

} // namespace graph
} // namespace omni_runtime
