#pragma once

#include <cstddef>
#include <limits>

namespace omni_runtime {
namespace graph {

struct PerfInfo {
  std::size_t run_count = 0U;
  double total_time_ms = 0.0;
  double minimum_time_ms = std::numeric_limits<double>::max();
  double maximum_time_ms = 0.0;

  double average_time_ms() const {
    return run_count == 0U ? 0.0 : total_time_ms / static_cast<double>(run_count);
  }
};

} // namespace graph
} // namespace omni_runtime
