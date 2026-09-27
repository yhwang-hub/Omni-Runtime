#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <variant>

#include "Omni-Runtime/backend/backend_compute_define.h"

namespace omni_runtime {
namespace inference {

struct OperatorGraphBindings final {
  using Value = std::variant<int32_t, uint64_t, float, bool, backend::DeviceView>;
  std::unordered_map<std::string, Value> values;
  std::function<bool(const std::string &, backend::DeviceView *const)> resolve_view;
};

using OperatorIterationBinder =
    std::function<bool(const std::string &, const int32_t, OperatorGraphBindings *const)>;

} // namespace inference
} // namespace omni_runtime
