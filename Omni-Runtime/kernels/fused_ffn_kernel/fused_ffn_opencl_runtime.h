#pragma once

#include "kernels/operator_runtime_common.h"

#include <limits>

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/opencl_operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::FusedFfn> final {
  static bool Make(const backend::FusedFfn &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const uint64_t local_bytes =
        (uint64_t(args.input_size > 0 ? args.input_size : 0) +
         uint64_t(args.intermediate_size > 0 ? args.intermediate_size : 0)) *
        sizeof(float);
    const size_t bytes = ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(
        args.rows <= 0 || args.input_size <= 0 || args.intermediate_size <= 0 ||
            args.output_size <= 0 || local_bytes > std::numeric_limits<int32_t>::max() ||
            !HasElements(args.input, uint64_t(args.rows) * args.input_size, bytes) ||
            !HasElements(args.fc1_weight, uint64_t(args.intermediate_size) * args.input_size,
                         bytes) ||
            !HasElements(args.fc1_bias, args.intermediate_size, bytes) ||
            !HasElements(args.fc2_weight, uint64_t(args.output_size) * args.intermediate_size,
                         bytes) ||
            !HasElements(args.fc2_bias, args.output_size, bytes) ||
            !HasElements(args.output, uint64_t(args.rows) * args.output_size, bytes),
        false, ERROR, "Invalid fused FFN dimensions, buffers, or local memory size");
    OMNI_RETURN_VAL_IF(
        !ConfigureOpenCL("fused_ffn_kernel/fused_ffn_kernel.cl", "FusedFfn", args.rows, 256U, call),
        false);
    BindOpenCLArguments(call, args.input, args.fc1_weight, args.fc1_bias, args.fc2_weight,
                        args.fc2_bias, args.rows, args.input_size, args.intermediate_size,
                        args.output_size, args.output);
    call->Local(static_cast<size_t>(local_bytes));
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
