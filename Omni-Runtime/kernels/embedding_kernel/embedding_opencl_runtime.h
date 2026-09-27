#pragma once

#include <tuple>

#include "Omni-Runtime/backend/backend_define.h"

#include "Omni-Runtime/backend/backend_opencl_utils.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::Embedding> final {
  static bool Make(const backend::Embedding &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const uint64_t values = uint64_t(args.token_count) * args.hidden_size;
    OMNI_RETURN_VAL_IF_LOG(
        args.token_count <= 0 || args.hidden_size <= 0 ||
            !backend::HasElements(args.indices, args.token_count, sizeof(int32_t)) ||
            !backend::HasElements(args.weight, args.hidden_size,
                                  backend::ElementBytes(call->type)) ||
            !backend::HasElements(args.output, values, backend::ElementBytes(call->type)),
        false, ERROR, "Invalid embedding operation");
    OMNI_RETURN_VAL_IF(
        !backend::opencl_internal::ConfigureOpenCL("embedding_kernel/embedding_kernel.cl",
                                                   "Embedding", (values + 255U) / 256U, 256U, call),
        false);
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::BindOpenCLArguments(
                           std::forward_as_tuple(args.indices, args.weight, args.token_count,
                                                 args.hidden_size, args.output),
                           call),
                       false);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
