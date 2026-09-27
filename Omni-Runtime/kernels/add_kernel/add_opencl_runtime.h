#pragma once

#include "kernels/operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::Add> final {
  static bool Make(const backend::Add &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const size_t bytes = ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(args.element_count == 0U ||
                               !HasElements(args.lhs, args.element_count, bytes) ||
                               !HasElements(args.rhs, args.element_count, bytes) ||
                               !HasElements(args.output, args.element_count, bytes),
                           false, ERROR, "Invalid add operation");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("add_kernel/add_kernel.cl", "Add",
                                        (args.element_count + 255U) / 256U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.lhs, args.rhs, args.element_count, args.output);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
