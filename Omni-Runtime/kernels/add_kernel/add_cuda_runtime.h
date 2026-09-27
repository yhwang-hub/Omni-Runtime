#pragma once

#include "Omni-Runtime/backend/backend_cuda_utils.h"
#include "kernels/kernel_runtime.h"

#include <cstdint>
#include <string>

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class AddKernelRuntime final {
public:
  struct Args final {
    backend::KernelLaunchConfig options;
    const void *const lhs;
    const void *const rhs;
    uint64_t element_count;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.lhs == nullptr || args.rhs == nullptr || args.output == nullptr, false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>("kernels/add_kernel/add_kernel.cuh",
                                               "omni_runtime::kernels::add_kernel::AddImpl"),
        args.options, static_cast<const T *>(args.lhs), static_cast<const T *>(args.rhs),
        args.element_count, static_cast<T *>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::Add> final {
  static bool Launch(const backend::Add &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.lhs.IsValid() || !args.rhs.IsValid() || !args.output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    return AddKernelRuntime<T>::CompileAndLaunch(
        "backend_Add",
        {.options = {.grid_dim = {static_cast<uint32_t>((args.element_count + 255U) / 256U), 1U,
                                  1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .lhs = backend::cuda_internal::CudaAddress<void>(args.lhs),
         .rhs = backend::cuda_internal::CudaAddress<void>(args.rhs),
         .element_count = args.element_count,
         .output = backend::cuda_internal::CudaAddress<void>(args.output)});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
