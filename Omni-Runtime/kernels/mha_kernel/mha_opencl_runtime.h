#pragma once

#include <tuple>

#include "Omni-Runtime/backend/backend_define.h"

#include <limits>

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "Omni-Runtime/backend/backend_opencl_utils.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::MultiHeadAttention> final {
  static bool Make(const backend::MultiHeadAttention &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(args.heads <= 0 || args.tokens <= 0 || args.head_dim <= 0 ||
                               args.head_dim > 256,
                           false, ERROR, "Invalid standalone MHA dimensions");
    const uint64_t rows = uint64_t(args.heads) * args.tokens;
    OMNI_RETURN_VAL_IF_LOG(rows > std::numeric_limits<uint64_t>::max() / args.head_dim, false,
                           ERROR, "Standalone MHA dimensions overflow");
    const uint64_t count = rows * args.head_dim;
    const size_t bytes = backend::ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(!backend::HasElements(args.query, count, bytes) ||
                               !backend::HasElements(args.key, count, bytes) ||
                               !backend::HasElements(args.value, count, bytes) ||
                               !backend::HasElements(args.output, count, bytes),
                           false, ERROR, "Invalid standalone MHA buffers");
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::ConfigureOpenCL(
                           "mha_kernel/mha_kernel.cl", "MultiHeadAttention", rows, 32U, call),
                       false);
    OMNI_RETURN_VAL_IF(
        !backend::opencl_internal::BindOpenCLArguments(
            std::forward_as_tuple(args.query, args.key, args.value, args.heads, args.tokens,
                                  args.head_dim, args.is_causal, args.output),
            call),
        false);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
