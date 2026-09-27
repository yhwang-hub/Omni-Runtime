#pragma once

#include <memory>

#include "Omni-Runtime/graph/basic/object.h"
#include "Omni-Runtime/graph/param/param_define.h"

namespace omni_runtime {
namespace graph {

class PassedParam : public Object {
public:
  PassedParam() = default;
  ~PassedParam() override = default;

  virtual std::shared_ptr<PassedParam> Clone() const = 0;

private:
  Status Run() final {
    return UnimplementedStatus("PassedParam cannot run");
  }
};

class DefaultPassedParam final : public PassedParam {
public:
  std::shared_ptr<PassedParam> Clone() const final {
    return std::make_shared<DefaultPassedParam>();
  }
};

} // namespace graph
} // namespace omni_runtime
