#pragma once

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/kernel_define.h"
#include "kernels/kernel_runtime.h"
#include "kernels/operator_runtime_common.h"

namespace omni_runtime::kernels::runtime {

template <typename T> T *CudaAddress(const backend::DeviceView &view) {
  if (!view.IsValid()) {
    return nullptr;
  }
  OMNI_RETURN_VAL_IF_LOG(view.backend_kind != backend::BackendKind::kCuda, nullptr, ERROR,
                         "A CUDA operator received memory from a different backend");
  return reinterpret_cast<T *>(static_cast<uint8_t *>(view.data) + view.offset_bytes);
}

template <typename T, typename Operation> struct CudaOperator;

} // namespace omni_runtime::kernels::runtime
