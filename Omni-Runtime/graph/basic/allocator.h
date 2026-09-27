#pragma once

#include <memory>
#include <type_traits>
#include <utility>

#include "Omni-Runtime/graph/basic/object.h"

namespace omni_runtime {
namespace graph {

class Allocator final {
public:
  template <typename T, typename... Args> static std::shared_ptr<T> MakeObject(Args &&...args) {
    static_assert(std::is_base_of<Object, T>::value, "T must inherit Object");
    return std::make_shared<T>(std::forward<Args>(args)...);
  }
};

} // namespace graph
} // namespace omni_runtime
