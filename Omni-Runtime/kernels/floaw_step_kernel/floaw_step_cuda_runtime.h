#pragma once

#include "kernels/operator_runtime_common.h"

#include "kernels/cuda_operator_runtime.h"
#include "Omni-Runtime/backend/backend_compute_define.h"

#include <algorithm>
#include <cmath>

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

template <typename T> struct CudaOperator<T, backend::FlowStep> final {
  static bool Launch(const backend::FlowStep &args) {
    OMNI_RETURN_VAL_IF_LOG(args.element_count == 0U || !std::isfinite(args.dt) ||
                               !HasElements(args.state, args.element_count, sizeof(T)) ||
                               !HasElements(args.velocity, args.element_count, sizeof(T)) ||
                               !HasElements(args.output, args.element_count, sizeof(T)),
                           false, ERROR, "Invalid CUDA flow integration operation");
    const uint32_t groups =
        static_cast<uint32_t>(std::min<uint64_t>((args.element_count - 1U) / 256U + 1U, 65535U));
    return FlowStepKernelRuntime<T>::CompileAndLaunch("backend_FlowStep",
                                                      {.options = {.grid_dim = {groups, 1U, 1U},
                                                                   .block_dim = {256U, 1U, 1U},
                                                                   .shared_memory_bytes = 0},
                                                       .state = CudaAddress<void>(args.state),
                                                       .velocity = CudaAddress<void>(args.velocity),
                                                       .dt = args.dt,
                                                       .element_count = args.element_count,
                                                       .output = CudaAddress<void>(args.output)});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
