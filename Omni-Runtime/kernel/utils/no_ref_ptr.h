#pragma once

namespace omni_runtime {
namespace jit {
namespace utils {

// The pointer itself is the kernel argument storage; do not take its address.
struct NoRefPtr {
  void *ptr = nullptr;
};

} // namespace utils
} // namespace jit
} // namespace omni_runtime
