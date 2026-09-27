#pragma once

#include "kernels/operator_runtime.h"

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class ArgmaxKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const logits;
    int32_t rows;
    int32_t columns;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.logits == nullptr || args.output == nullptr, false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>("kernels/argmax_kernel/argmax_kernel.cuh",
                                               "omni_runtime::kernels::argmax_kernel::ArgmaxImpl"),
        args.options, static_cast<const T *>(args.logits), args.rows, args.columns,
        static_cast<int32_t *>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::Argmax> final {
  static bool Launch(const backend::Argmax &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.logits.IsValid() || !args.output.IsValid(), false, ERROR,
                           "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.columns <= 0, false, ERROR,
                           "Invalid compute operation dimensions");
    return ArgmaxKernelRuntime<T>::CompileAndLaunch(
        "backend_Argmax", {.options = {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
                                       .block_dim = {256U, 1U, 1U},
                                       .shared_memory_bytes = 0},
                           .logits = CudaAddress<void>(args.logits),
                           .rows = args.rows,
                           .columns = args.columns,
                           .output = CudaAddress<void>(args.output)});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
