#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "Omni-Runtime/graph/daemon/daemon.h"

namespace omni_runtime {
namespace graph {

class Storage;

class DaemonManager final : public Object {
public:
  DaemonManager() = default;
  ~DaemonManager() override;

  Status Add(const std::shared_ptr<Daemon> &daemon);
  Status Remove(const std::shared_ptr<Daemon> &daemon);
  bool SetContext(const std::shared_ptr<ParamManager> &param_manager,
                  const std::shared_ptr<EventManager> &event_manager);
  std::size_t size() const;
  Status Clear();

  Status Init() final;
  Status Destroy() final;

private:
  Status Run() final {
    return UnimplementedStatus("DaemonManager cannot run");
  }

  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<Daemon>> daemons_;
  std::shared_ptr<ParamManager> param_manager_;
  std::shared_ptr<EventManager> event_manager_;

  friend class Storage;
};

} // namespace graph
} // namespace omni_runtime
