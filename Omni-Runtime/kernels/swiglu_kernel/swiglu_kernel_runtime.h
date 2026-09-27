#pragma once

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class SwigluKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const gate;
    const void *const up;
    uint64_t element_count;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.gate == nullptr || args.up == nullptr || args.output == nullptr, false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>("kernels/swiglu_kernel/swiglu_kernel.cuh",
                                               "omni_runtime::kernels::swiglu_kernel::SwigluImpl"),
        args.options, static_cast<const T *>(args.gate), static_cast<const T *>(args.up),
        args.element_count, static_cast<T *>(args.output));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
