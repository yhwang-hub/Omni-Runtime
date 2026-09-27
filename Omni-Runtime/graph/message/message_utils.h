#pragma once

#include "Omni-Runtime/graph/message/message_manager.h"

namespace omni_runtime {
namespace graph {

inline MessageManager *GlobalMessageManager() {
  return MessageManager::Instance();
}

} // namespace graph
} // namespace omni_runtime
