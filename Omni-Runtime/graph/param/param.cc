#include "Omni-Runtime/graph/param/param.h"

#include <algorithm>

#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace graph {

Status Param::AddBacktrace(const std::string &trace) {
  OMNI_RETURN_VAL_IF_LOG(!enable_backtrace_,
                    InvalidArgumentStatus("backtrace is disabled for param " + key_), ERROR,
                    "backtrace is disabled for param " << key_);
  std::lock_guard<std::mutex> lock(backtrace_mutex_);
  if (std::find(backtrace_.begin(), backtrace_.end(), trace) == backtrace_.end()) {
    backtrace_.emplace_back(trace);
  }
  return Status();
}

Status Param::GetBacktrace(std::vector<std::string> *const traces) const {
  OMNI_RETURN_VAL_IF_LOG(traces == nullptr, InvalidArgumentStatus("traces is null"), ERROR,
                    "traces is null");
  OMNI_RETURN_VAL_IF_LOG(!enable_backtrace_,
                    InvalidArgumentStatus("backtrace is disabled for param " + key_), ERROR,
                    "backtrace is disabled for param " << key_);
  std::lock_guard<std::mutex> lock(backtrace_mutex_);
  *traces = backtrace_;
  return Status();
}

bool Param::CleanBacktrace() {
  if (!enable_backtrace_) {
    return true;
  }
  std::lock_guard<std::mutex> lock(backtrace_mutex_);
  backtrace_.clear();
  return true;
}

} // namespace graph
} // namespace omni_runtime
