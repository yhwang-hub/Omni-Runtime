#pragma once

#include <cstddef>
#include <cstdint>

namespace omni_runtime {
namespace graph {

using Index = int32_t;
using Level = int32_t;
using Milliseconds = int64_t;

constexpr std::size_t kDefaultLoopCount = 1U;
constexpr Level kDefaultElementLevel = 0;
constexpr Milliseconds kNoTimeoutMs = 0;
constexpr Milliseconds kMaxBlockingTimeoutMs = 1000L * 3600L * 24L;

} // namespace graph
} // namespace omni_runtime
