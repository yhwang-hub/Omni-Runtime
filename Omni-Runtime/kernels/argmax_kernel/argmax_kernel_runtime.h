#pragma once

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

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
