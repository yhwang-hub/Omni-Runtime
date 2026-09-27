#pragma once

#include <cstdint>

namespace omni_runtime {
namespace graph {

enum class EventType : uint8_t {
  kSync = 0,
  kAsync = 1,
};

enum class EventAsyncStrategy : uint8_t {
  kPipelineRunFinish = 0,
  kPipelineDestroy = 1,
  kNoWait = 2,
};

} // namespace graph
} // namespace omni_runtime
