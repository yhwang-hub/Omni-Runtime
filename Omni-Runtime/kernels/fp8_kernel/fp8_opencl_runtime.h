#pragma once

#include <tuple>

#include "Omni-Runtime/backend/backend_opencl_utils.h"
#include "kernels/fp8_kernel/fp8_runtime_utils.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::Fp8Linear> final {
  static bool Make(const backend::Fp8Linear &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF_LOG(call == nullptr, false, ERROR, "FP8 invocation output is null");
    OMNI_RETURN_VAL_IF_LOG(!ValidateFp8(args, backend::ElementBytes(call->type)), false, ERROR,
                           "Invalid FP8 E4M3FN block-linear buffers or dimensions");
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::ConfigureOpenCL(
                           "fp8_kernel/fp8_kernel.cl", "Fp8Linear",
                           (static_cast<uint64_t>(args.rows) * args.columns + 63U) / 64U, 64U,
                           call),
                       false);
    return backend::opencl_internal::BindOpenCLArguments(
        std::forward_as_tuple(args.input, args.weight, args.weight_scales, args.rows, args.inner,
                              args.columns, args.block_reduction_groups, args.output),
        call);
  }
};

} // namespace omni_runtime::kernels::runtime
