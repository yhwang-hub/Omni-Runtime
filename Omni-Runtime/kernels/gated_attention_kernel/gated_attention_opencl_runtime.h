#pragma once

#include "kernels/operator_runtime_common.h"

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/opencl_operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::SplitGatedQuery> final {
  static bool Make(const backend::SplitGatedQuery &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.head_count <= 0 || args.head_size <= 0,
                           false, ERROR, "Invalid gated query dimensions");
    const uint64_t rows = uint64_t(args.token_count) * args.head_count;
    OMNI_RETURN_VAL_IF_LOG(rows > std::numeric_limits<uint64_t>::max() /
                                      (2U * uint64_t(args.head_size)),
                           false, ERROR, "Gated query dimensions overflow");
    const uint64_t count = rows * args.head_size;
    const size_t bytes = ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(!HasElements(args.query_gate, 2U * count, bytes) ||
                               !HasElements(args.query, count, bytes) ||
                               !HasElements(args.gate, count, bytes),
                           false, ERROR, "Invalid gated query buffers");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("gated_attention_kernel/gated_attention_kernel.cl",
                                        "SplitGatedQuery", (count - 1U) / 256U + 1U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.query_gate, args.token_count, args.head_count, args.head_size,
                        args.query, args.gate);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
