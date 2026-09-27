#pragma once

#include "Omni-Runtime/graph/basic/status.h"

namespace omni_runtime {
namespace graph {

class Object {
public:
  Object() = default;
  virtual ~Object() = default;

  Object(const Object &) = delete;
  Object &operator=(const Object &) = delete;

  virtual Status Init() {
    return Status();
  }

  virtual Status Run() = 0;

  virtual Status Destroy() {
    return Status();
  }
};

} // namespace graph
} // namespace omni_runtime
