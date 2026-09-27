#pragma once

namespace omni_runtime {
namespace jit {
namespace utils {

class GilScopedRelease final {
public:
  GilScopedRelease() = default;
  GilScopedRelease(const GilScopedRelease &) = delete;
  GilScopedRelease &operator=(const GilScopedRelease &) = delete;
};

} // namespace utils
} // namespace jit
} // namespace omni_runtime
