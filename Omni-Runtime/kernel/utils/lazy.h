#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <dlfcn.h>

#include "Omni-Runtime/kernel/utils/exception.h"

#define JIT_DECLARE_STATIC_VAR_IN_CLASS(cls, name) decltype(cls::name) cls::name

namespace omni_runtime {
namespace jit {
namespace utils {
namespace lazy {

template <typename T> class LazyInit {
  std::shared_ptr<T> ptr;
  std::function<std::shared_ptr<T>()> factory;

public:
  explicit LazyInit(std::nullptr_t) {
  }

  explicit LazyInit(std::function<std::shared_ptr<T>()> factory) : factory(std::move(factory)) {
  }

  T *operator->() {
    JIT_HOST_ASSERT(factory != nullptr, "lazy object must be initialized before use");
    if (ptr == nullptr) {
      ptr = factory();
    }
    JIT_HOST_ASSERT(ptr != nullptr, "lazy factory must not return nullptr");
    return ptr.get();
  }

  std::shared_ptr<T> get() {
    (void)operator->();
    return ptr;
  }
};

} // namespace lazy
} // namespace utils
} // namespace jit
} // namespace omni_runtime

#define JIT_DECL_LAZY_DL_HANDLE(handle_func_name, lib)                                             \
  inline void *handle_func_name() {                                                                \
    static void *handle = [] {                                                                     \
      ::dlerror();                                                                                 \
      void *value = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);                                           \
      if (value == nullptr) {                                                                      \
        const char *error = ::dlerror();                                                           \
        JIT_PANIC("failed to load {}: {}", lib, error == nullptr ? "unknown" : error);             \
      }                                                                                            \
      return value;                                                                                \
    }();                                                                                           \
    return handle;                                                                                 \
  }

#define JIT_DECL_LAZY_DL_FUNCTION(handle_func_name, name)                                          \
  template <typename... Args> static auto lazy_##name(Args &&...args) {                            \
    static const auto func = []() {                                                                \
      void *symbol = ::dlsym(handle_func_name(), #name);                                           \
      if (symbol == nullptr) {                                                                     \
        const char *error = ::dlerror();                                                           \
        JIT_PANIC("failed to load {} from {}: {}", #name, #handle_func_name,                       \
                  error == nullptr ? "unknown" : error);                                           \
      }                                                                                            \
      return reinterpret_cast<decltype(&name)>(symbol);                                            \
    }();                                                                                           \
    return func(std::forward<Args>(args)...);                                                      \
  }
