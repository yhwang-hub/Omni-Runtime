#pragma once

#include "kernels/operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::Embedding> final {
  static bool Make(const backend::Embedding &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const uint64_t values = uint64_t(args.token_count) * args.hidden_size;
    OMNI_RETURN_VAL_IF_LOG(
        args.token_count <= 0 || args.hidden_size <= 0 ||
            !HasElements(args.indices, args.token_count, sizeof(int32_t)) ||
            !HasElements(args.weight, args.hidden_size, ElementBytes(call->type)) ||
            !HasElements(args.output, values, ElementBytes(call->type)),
        false, ERROR, "Invalid embedding operation");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("embedding_kernel/embedding_kernel.cl", "Embedding",
                                        (values + 255U) / 256U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.indices, args.weight, args.token_count, args.hidden_size,
                        args.output);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
