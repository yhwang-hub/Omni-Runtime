#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "Omni-Runtime/utils/logger.h"

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
  std::size_t offset_bytes = 0;
};

struct DeviceView final {
  void *data = nullptr;
  std::size_t offset_bytes = 0;
  std::size_t size_bytes = 0;
  BackendKind backend_kind = BackendKind::kAuto;

  bool IsValid() const {
    return data != nullptr && size_bytes > 0U &&
           offset_bytes <= std::numeric_limits<std::size_t>::max() - size_bytes;
  }

  DeviceView Slice(const std::size_t offset, const std::size_t bytes) const {
    OMNI_RETURN_VAL_IF_LOG(!IsValid() || bytes == 0U || offset > size_bytes ||
                               bytes > size_bytes - offset,
                           DeviceView{}, ERROR, "Invalid device view slice");
    return {.data = data,
            .offset_bytes = offset_bytes + offset,
            .size_bytes = bytes,
            .backend_kind = backend_kind};
  }
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
