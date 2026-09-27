#pragma once

#include "Omni-Runtime/backend/backend_cuda_utils.h"
#include "kernels/kernel_runtime.h"

#include <cstdint>
#include <string>

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class SwigluKernelRuntime final {
public:
  struct Args final {
    backend::KernelLaunchConfig options;
    const void *const gate;
    const void *const up;
    uint64_t element_count;
    void *const output;
    bool is_ggml_silu = false;
    bool is_float_gate = false;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.gate == nullptr || args.up == nullptr || args.output == nullptr, false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        args.is_float_gate ? CudaKernelRuntime::MakeKernelSource<T, 2>(
                                 "kernels/swiglu_kernel/swiglu_kernel.cuh",
                                 "omni_runtime::kernels::swiglu_kernel::SwigluImpl")
        : args.is_ggml_silu ? CudaKernelRuntime::MakeKernelSource<T, 1>(
                                  "kernels/swiglu_kernel/swiglu_kernel.cuh",
                                  "omni_runtime::kernels::swiglu_kernel::SwigluImpl")
                            : CudaKernelRuntime::MakeKernelSource<T, 0>(
                                  "kernels/swiglu_kernel/swiglu_kernel.cuh",
                                  "omni_runtime::kernels::swiglu_kernel::SwigluImpl"),
        args.options, static_cast<const T *>(args.gate), static_cast<const T *>(args.up),
        args.element_count, static_cast<T *>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::SwiGLU> final {
  static bool Launch(const backend::SwiGLU &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.gate.IsValid() || !args.up.IsValid() || !args.output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    return SwigluKernelRuntime<T>::CompileAndLaunch(
        "backend_SwiGLU",
        {.options = {.grid_dim = {static_cast<uint32_t>((args.element_count + 255U) / 256U), 1U,
                                  1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .gate = backend::cuda_internal::CudaAddress<void>(args.gate),
         .up = backend::cuda_internal::CudaAddress<void>(args.up),
         .element_count = args.element_count,
         .output = backend::cuda_internal::CudaAddress<void>(args.output),
         .is_ggml_silu = args.is_ggml_silu,
         .is_float_gate = args.is_float_gate});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
