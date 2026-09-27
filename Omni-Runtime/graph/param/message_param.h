#pragma once

#include "Omni-Runtime/graph/basic/object.h"

namespace omni_runtime {
namespace graph {

class MessageParam : public Object {
public:
  MessageParam() = default;
  ~MessageParam() override = default;

private:
  Status Run() final {
    return UnimplementedStatus("MessageParam cannot run");
  }
};

} // namespace graph
} // namespace omni_runtime
