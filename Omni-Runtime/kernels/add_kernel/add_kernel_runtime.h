#pragma once

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class AddKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
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

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
