#pragma once

#include <cstddef>
#include <cstdint>

namespace omni_runtime {
namespace backend {

enum class BackendKind : uint8_t { kCuda = 0, kOpenCL = 1, kAuto = 2 };

enum class ScalarType : uint8_t { kFloat32 = 0, kFloat16 = 1, kBFloat16 = 2 };

// Backend-specific flags; zero requests each backend's default behavior.
inline constexpr uint32_t kDefaultStreamFlags = 0U;
inline constexpr uint32_t kDefaultEventFlags = 0U;
inline constexpr uint32_t kDefaultEventWaitFlags = 0U;

enum class MemoryKind : uint8_t {
  kHost = 0,
  kHostView = 1,
  kPinnedHost = 2,
  kManaged = 3,
  kDevice = 4,
  kDeviceView = 5,
};

struct Allocation final {
  void *data = nullptr;
  std::size_t size_bytes = 0;
  MemoryKind memory_kind = MemoryKind::kHost;
};

struct EventHandle final {
  void *native_handle = nullptr;
  BackendKind backend_kind = BackendKind::kCuda;
};

struct StreamHandle final {
  void *native_handle = nullptr;
  BackendKind backend_kind = BackendKind::kCuda;
};

} // namespace backend
} // namespace omni_runtime
