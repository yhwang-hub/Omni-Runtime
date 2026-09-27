#pragma once

#include "kernels/operator_runtime_common.h"

#include "kernels/opencl_operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::SwiGLU> final {
  static bool Make(const backend::SwiGLU &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const size_t bytes = ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(args.element_count == 0U ||
                               !HasElements(args.gate, args.element_count, bytes) ||
                               !HasElements(args.up, args.element_count, bytes) ||
                               !HasElements(args.output, args.element_count, bytes),
                           false, ERROR, "Invalid SwiGLU operation");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("swiglu_kernel/swiglu_kernel.cl", "SwiGLU",
                                        (args.element_count + 255U) / 256U, 256U, call),
                       false);
    if (args.is_ggml_silu) {
      call->build_defines.push_back("OMNI_GGML_SILU");
    }
    BindOpenCLArguments(call, args.gate, args.up, args.element_count, args.output);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
