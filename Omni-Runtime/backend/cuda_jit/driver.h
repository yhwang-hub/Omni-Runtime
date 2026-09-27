#pragma once

#include <cuda.h>

#include "Omni-Runtime/kernel/utils/exception.h"
#include "Omni-Runtime/kernel/utils/lazy.h"

JIT_STATIC_ASSERT(CUDA_VERSION >= 12040, "DeepJIT requires CUDA 12.4 or newer");

namespace omni_runtime {
namespace jit {
namespace backend {
namespace cuda {
namespace driver {

JIT_DECL_LAZY_DL_HANDLE(get_cuda_handle, "libcuda.so.1");

JIT_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuGetErrorName);
JIT_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuGetErrorString);
JIT_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuFuncSetAttribute);
JIT_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLaunchKernelEx);
JIT_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuTensorMapEncodeTiled);
JIT_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryLoadFromFile)
JIT_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryUnload)
JIT_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryGetKernelCount)
JIT_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryEnumerateKernels)
JIT_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuKernelGetFunction)

inline void check_cuda_driver(const CUresult error, const char *expression) {
  if (error == CUDA_SUCCESS) {
    return;
  }

  const char *error_name = "unknown";
  const char *error_description = "unknown";
  lazy_cuGetErrorName(error, &error_name);
  lazy_cuGetErrorString(error, &error_description);
  JIT_PANIC("{} failed with CUDA error {} ({}): {}", expression, static_cast<int>(error),
            error_name, error_description);
}

#ifndef JIT_CUDA_DRIVER_CHECK
#define JIT_CUDA_DRIVER_CHECK(expr)                                                                \
  ::omni_runtime::jit::backend::cuda::driver::check_cuda_driver((expr), #expr)
#endif

} // namespace driver
} // namespace cuda
} // namespace backend
} // namespace jit
} // namespace omni_runtime
