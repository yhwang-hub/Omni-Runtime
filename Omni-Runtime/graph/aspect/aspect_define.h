#pragma once

#include <cstdint>

namespace omni_runtime {
namespace graph {

enum class AspectType : uint8_t {
  kBeginInit = 0,
  kFinishInit = 1,
  kBeginRun = 2,
  kFinishRun = 3,
  kBeginDestroy = 4,
  kFinishDestroy = 5,
  kEnterTimeout = 6,
  kEnterCrashed = 7,
};

} // namespace graph
} // namespace omni_runtime
