#pragma once

#include <cstdint>

namespace omni_runtime {
namespace graph {

enum class MessagePushStrategy : uint8_t {
  kWait = 0,
  kReplace = 1,
  kDrop = 2,
};

using ConnectionId = int64_t;
constexpr ConnectionId kInvalidConnectionId = -1;

} // namespace graph
} // namespace omni_runtime
