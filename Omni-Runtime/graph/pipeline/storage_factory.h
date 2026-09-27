#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>

#include "Omni-Runtime/graph/basic/object.h"

namespace omni_runtime {
namespace graph {

class StorageFactory final {
public:
  template <typename T> static bool Register() {
    static_assert(std::is_base_of<Object, T>::value, "T must inherit Object");
    static_assert(std::is_default_constructible<T>::value,
                  "T must be default constructible for storage recovery");
    std::lock_guard<std::mutex> lock(mutex_);
    creators_[typeid(T).name()] = []() {
      return std::static_pointer_cast<Object>(std::make_shared<T>());
    };
    return true;
  }

  static std::shared_ptr<Object> Create(const std::string &type_name);
  static bool Has(const std::string &type_name);
  static bool Clear();

private:
  static bool RegisterBuiltinTypes();

  static std::mutex mutex_;
  static std::unordered_map<std::string, std::function<std::shared_ptr<Object>()>> creators_;
};

} // namespace graph
} // namespace omni_runtime
