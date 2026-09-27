#pragma once

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class GatedAddKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const lhs;
    const void *const rhs;
    const void *const gate;
    uint64_t element_count;
    int32_t hidden_size;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.lhs == nullptr || args.rhs == nullptr || args.gate == nullptr ||
                           args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/gated_add_kernel/gated_add_kernel.cuh",
            "omni_runtime::kernels::gated_add_kernel::GatedAddImpl"),
        args.options, static_cast<const T *>(args.lhs), static_cast<const T *>(args.rhs),
        static_cast<const T *>(args.gate), args.element_count, args.hidden_size,
        static_cast<T *>(args.output));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
