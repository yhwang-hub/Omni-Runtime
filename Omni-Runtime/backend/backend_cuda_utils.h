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

template <typename T> T *CudaAddress(const DeviceView &view) {
  if (!view.IsValid()) {
    return nullptr;
  }
  OMNI_RETURN_VAL_IF_LOG(view.backend_kind != BackendKind::kCuda, nullptr, ERROR,
                         "A CUDA operator received memory from a different backend");
  return reinterpret_cast<T *>(static_cast<uint8_t *>(view.data) + view.offset_bytes);
}

} // namespace omni_runtime::backend::cuda_internal
