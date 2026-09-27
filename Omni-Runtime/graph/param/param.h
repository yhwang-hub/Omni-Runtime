#pragma once

#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "Omni-Runtime/graph/basic/object.h"

namespace omni_runtime {
namespace graph {

class Storage;

class Param : public Object {
public:
  Param() = default;
  ~Param() override = default;

  Status AddBacktrace(const std::string &trace);
  Status GetBacktrace(std::vector<std::string> *const traces) const;
  bool CleanBacktrace();

  const std::string &key() const {
    return key_;
  }

  void Lock() {
    value_mutex_.lock();
  }

  void Unlock() {
    value_mutex_.unlock();
  }

  bool TryLock() {
    return value_mutex_.try_lock();
  }

  virtual Status Setup() {
    return Status();
  }

  virtual bool Reset(const Status &status) {
    static_cast<void>(status);
    return true;
  }

protected:
  std::shared_mutex &value_mutex() {
    return value_mutex_;
  }

private:
  Status Run() final {
    return UnimplementedStatus("Param cannot run");
  }

  bool enable_backtrace_ = false;
  std::string key_;
  mutable std::mutex backtrace_mutex_;
  std::vector<std::string> backtrace_;
  std::shared_mutex value_mutex_;

  friend class ParamManager;
  friend class Storage;
};

} // namespace graph
} // namespace omni_runtime
