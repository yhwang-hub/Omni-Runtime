#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "Omni-Runtime/graph/aspect/aspect.h"

namespace omni_runtime {
namespace graph {

class Storage;

class AspectManager final {
public:
  AspectManager() = default;
  ~AspectManager() = default;

  Status Add(const std::shared_ptr<Aspect> &aspect, const AspectParam *const param);
  Status Reflect(const AspectType type, const Status &current_status) const;
  Status PopLast();
  bool Clear();
  std::size_t size() const;
  bool SetContext(const std::shared_ptr<ParamManager> &param_manager,
                  const std::shared_ptr<EventManager> &event_manager,
                  const std::string &belong_name);

private:
  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<Aspect>> aspects_;
  std::shared_ptr<ParamManager> param_manager_;
  std::shared_ptr<EventManager> event_manager_;
  std::string belong_name_;

  friend class Storage;
};

} // namespace graph
} // namespace omni_runtime
