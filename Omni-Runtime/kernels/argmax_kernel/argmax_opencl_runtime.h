#pragma once

#include <tuple>

#include "Omni-Runtime/backend/backend_define.h"

#include "Omni-Runtime/backend/backend_opencl_utils.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::Argmax> final {
  static bool Make(const backend::Argmax &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.columns <= 0 ||
                               !backend::HasElements(args.logits,
                                                     uint64_t(args.rows) * args.columns,
                                                     backend::ElementBytes(call->type)) ||
                               !backend::HasElements(args.output, args.rows, sizeof(int32_t)),
                           false, ERROR, "Invalid argmax operation");
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::ConfigureOpenCL("argmax_kernel/argmax_kernel.cl",
                                                                  "Argmax", args.rows, 256U, call),
                       false);
    OMNI_RETURN_VAL_IF(
        !backend::opencl_internal::BindOpenCLArguments(
            std::forward_as_tuple(args.logits, args.rows, args.columns, args.output), call),
        false);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
