#pragma once

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class FlowStepKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const state;
    const void *const velocity;
    float dt;
    uint64_t element_count;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.state == nullptr || args.velocity == nullptr || args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/floaw_step_kernel/flow_step_kernel.cuh",
            "omni_runtime::kernels::flow_step_kernel::FlowStepImpl"),
        args.options, static_cast<const T *>(args.state), static_cast<const T *>(args.velocity),
        args.dt, args.element_count, static_cast<T *>(args.output));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
