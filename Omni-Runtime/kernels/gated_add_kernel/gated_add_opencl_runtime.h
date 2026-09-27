#pragma once

#include <tuple>

#include "Omni-Runtime/backend/backend_define.h"

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "Omni-Runtime/backend/backend_opencl_utils.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::GatedAdd> final {
  static bool Make(const backend::GatedAdd &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const size_t bytes = backend::ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(args.element_count == 0U || args.hidden_size <= 0 ||
                               !backend::HasElements(args.lhs, args.element_count, bytes) ||
                               !backend::HasElements(args.rhs, args.element_count, bytes) ||
                               !backend::HasElements(args.gate, args.hidden_size, bytes) ||
                               !backend::HasElements(args.output, args.element_count, bytes),
                           false, ERROR, "Invalid gated addition operation");
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::ConfigureOpenCL(
                           "gated_add_kernel/gated_add_kernel.cl", "GatedAdd",
                           (args.element_count - 1U) / 256U + 1U, 256U, call),
                       false);
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::BindOpenCLArguments(
                           std::forward_as_tuple(args.lhs, args.rhs, args.gate, args.element_count,
                                                 args.hidden_size, args.output),
                           call),
                       false);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
