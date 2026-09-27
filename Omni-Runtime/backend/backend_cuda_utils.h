#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "Omni-Runtime/backend/backend_define.h"

namespace omni_runtime::backend::cuda_internal {

inline cudaEvent_t NativeEvent(const void *event) {
  return reinterpret_cast<cudaEvent_t>(const_cast<void *>(event));
}

inline cudaStream_t NativeStream(const void *stream) {
  return reinterpret_cast<cudaStream_t>(const_cast<void *>(stream));
}

inline void *AllocationAddress(const Allocation &allocation) {
  return static_cast<uint8_t *>(allocation.data) + allocation.offset_bytes;
}

} // namespace omni_runtime::backend::cuda_internal
