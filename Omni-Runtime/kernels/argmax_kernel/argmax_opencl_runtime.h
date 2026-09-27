#pragma once

#include "kernels/operator_runtime_common.h"

#include "kernels/opencl_operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::Argmax> final {
  static bool Make(const backend::Argmax &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.columns <= 0 ||
                               !HasElements(args.logits, uint64_t(args.rows) * args.columns,
                                            ElementBytes(call->type)) ||
                               !HasElements(args.output, args.rows, sizeof(int32_t)),
                           false, ERROR, "Invalid argmax operation");
    OMNI_RETURN_VAL_IF(
        !ConfigureOpenCL("argmax_kernel/argmax_kernel.cl", "Argmax", args.rows, 256U, call), false);
    BindOpenCLArguments(call, args.logits, args.rows, args.columns, args.output);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
