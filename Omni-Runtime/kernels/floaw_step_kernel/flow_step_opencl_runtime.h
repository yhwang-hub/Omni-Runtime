#pragma once

#include <tuple>

#include "Omni-Runtime/backend/backend_define.h"

#include <cmath>

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "Omni-Runtime/backend/backend_opencl_utils.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::FlowStep> final {
  static bool Make(const backend::FlowStep &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const size_t bytes = backend::ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(args.element_count == 0U || !std::isfinite(args.dt) ||
                               !backend::HasElements(args.state, args.element_count, bytes) ||
                               !backend::HasElements(args.velocity, args.element_count, bytes) ||
                               !backend::HasElements(args.output, args.element_count, bytes),
                           false, ERROR, "Invalid flow integration operation");
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::ConfigureOpenCL(
                           "floaw_step_kernel/flow_step_kernel.cl", "FlowStep",
                           (args.element_count - 1U) / 256U + 1U, 256U, call),
                       false);
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::BindOpenCLArguments(
                           std::forward_as_tuple(args.state, args.velocity, args.dt,
                                                 args.element_count, args.output),
                           call),
                       false);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
