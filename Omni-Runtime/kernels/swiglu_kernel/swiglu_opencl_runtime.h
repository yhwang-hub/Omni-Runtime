#pragma once

#include <tuple>

#include "Omni-Runtime/backend/backend_define.h"

#include "Omni-Runtime/backend/backend_opencl_utils.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::SwiGLU> final {
  static bool Make(const backend::SwiGLU &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const size_t bytes = backend::ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(args.element_count == 0U ||
                               !backend::HasElements(args.gate, args.element_count, bytes) ||
                               !backend::HasElements(args.up, args.element_count, bytes) ||
                               !backend::HasElements(args.output, args.element_count, bytes),
                           false, ERROR, "Invalid SwiGLU operation");
    OMNI_RETURN_VAL_IF(
        !backend::opencl_internal::ConfigureOpenCL("swiglu_kernel/swiglu_kernel.cl", "SwiGLU",
                                                   (args.element_count + 255U) / 256U, 256U, call),
        false);
    if (args.is_ggml_silu) {
      call->build_defines.push_back("OMNI_GGML_SILU");
    }
    if (args.is_float_gate) {
      call->build_defines.push_back("OMNI_FLOAT_GATE");
    }
    OMNI_RETURN_VAL_IF(
        !backend::opencl_internal::BindOpenCLArguments(
            std::forward_as_tuple(args.gate, args.up, args.element_count, args.output), call),
        false);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
