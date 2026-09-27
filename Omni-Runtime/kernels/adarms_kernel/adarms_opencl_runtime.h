#pragma once

#include "kernels/operator_runtime_common.h"

#include <cmath>

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/opencl_operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::AdaRmsNorm> final {
  static bool Make(const backend::AdaRmsNorm &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const size_t bytes = ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(
        args.rows <= 0 || args.hidden_size <= 0 || !std::isfinite(args.epsilon) ||
            args.epsilon <= 0.0F ||
            !HasElements(args.input, uint64_t(args.rows) * args.hidden_size, bytes) ||
            !HasElements(args.scale, args.hidden_size, bytes) ||
            !HasElements(args.shift, args.hidden_size, bytes) ||
            !HasElements(args.output, uint64_t(args.rows) * args.hidden_size, bytes),
        false, ERROR, "Invalid adaptive RMS normalization operation");
    OMNI_RETURN_VAL_IF(
        !ConfigureOpenCL("adarms_kernel/adarms_kernel.cl", "AdaRmsNorm", args.rows, 256U, call),
        false);
    BindOpenCLArguments(call, args.input, args.scale, args.shift, args.rows, args.hidden_size,
                        args.epsilon, args.output);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
