#pragma once

#include <cstdint>

namespace omni_runtime {
namespace graph {

struct PipelineConfig {
  uint32_t thread_count = 1U;
};

} // namespace graph
} // namespace omni_runtime
