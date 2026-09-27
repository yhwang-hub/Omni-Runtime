#pragma once

#include <tuple>

#include "Omni-Runtime/backend/backend_define.h"

#include "Omni-Runtime/backend/backend_opencl_utils.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::Add> final {
  static bool Make(const backend::Add &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const size_t bytes = backend::ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(args.element_count == 0U ||
                               !backend::HasElements(args.lhs, args.element_count, bytes) ||
                               !backend::HasElements(args.rhs, args.element_count, bytes) ||
                               !backend::HasElements(args.output, args.element_count, bytes),
                           false, ERROR, "Invalid add operation");
    OMNI_RETURN_VAL_IF(
        !backend::opencl_internal::ConfigureOpenCL("add_kernel/add_kernel.cl", "Add",
                                                   (args.element_count + 255U) / 256U, 256U, call),
        false);
    OMNI_RETURN_VAL_IF(
        !backend::opencl_internal::BindOpenCLArguments(
            std::forward_as_tuple(args.lhs, args.rhs, args.element_count, args.output), call),
        false);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
