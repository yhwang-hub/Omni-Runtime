#pragma once

#include <memory>
#include <unordered_map>

namespace omni_runtime {
namespace jit {
namespace backend {
namespace memory {

// In-memory cache keyed by value type.
template <typename Key, typename Value> class MemCache {
public:
  std::unordered_map<Key, std::shared_ptr<Value>> cache;

  template <typename Factory>
  std::shared_ptr<Value> get_or_create(const Key &key, Factory &&factory) {
    if (const auto iterator = cache.find(key); iterator != cache.end()) {
      return iterator->second;
    }
    auto value = factory();
    cache.emplace(key, value);
    return value;
  }
};

} // namespace memory
} // namespace backend
} // namespace jit
} // namespace omni_runtime
