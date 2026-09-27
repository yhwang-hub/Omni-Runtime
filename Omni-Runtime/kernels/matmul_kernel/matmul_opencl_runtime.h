#pragma once

#include "kernels/operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::Linear> final {
  static bool Make(const backend::Linear &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const size_t bytes = ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(
        args.rows <= 0 || args.inner <= 0 || args.columns <= 0 ||
            !HasElements(args.input, uint64_t(args.rows) * args.inner, bytes) ||
            !HasElements(args.weight, uint64_t(args.columns) * args.inner, bytes) ||
            !HasElements(args.output, uint64_t(args.rows) * args.columns, bytes),
        false, ERROR, "Invalid linear operation");
    OMNI_RETURN_VAL_IF(
        !ConfigureOpenCL("matmul_kernel/matmul_kernel.cl", "Linear", args.columns, 256U, call),
        false);
    if (args.rows > 1) {
      call->dimensions = 2U;
      call->local_size = {16U, 16U, 1U};
      call->global_size = {RoundWorkSize(args.columns, 16U), RoundWorkSize(args.rows, 16U), 1U};
    }
    if (call->has_nvidia_mma && (call->type == backend::ScalarType::kFloat16 ||
                                 call->type == backend::ScalarType::kBFloat16)) {
      if (args.rows > 1) {
        const size_t tile_rows =
            args.rows >= 1024 && args.inner >= 256 && args.inner % 8 == 0 && args.columns >= 256 &&
                    args.input.offset_bytes % 16U == 0U && args.weight.offset_bytes % 16U == 0U
                ? 128U
                : 64U;
        call->entry = "LinearTensor";
        call->dimensions = 2U;
        call->local_size = {tile_rows * 2U, 1U, 1U};
        call->global_size = {RoundWorkSize(args.rows, tile_rows) / tile_rows * tile_rows * 2U,
                             RoundWorkSize(args.columns, 64U) / 64U, 1U};
        if (tile_rows == 128U) {
          call->build_defines.push_back("OMNI_LINEAR_TILE=1");
        }
      } else {
        call->entry = "LinearVector";
        call->dimensions = 1U;
        call->local_size = {256U, 1U, 1U};
        call->global_size = {RoundWorkSize(args.columns, 8U) / 8U * 256U, 1U, 1U};
      }
    }
    BindOpenCLArguments(call, args.input, args.weight, args.rows, args.inner, args.columns,
                        args.output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::ParallelLinear> final {
  static bool Make(const backend::ParallelLinear &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF_LOG(call == nullptr, false, ERROR, "Parallel invocation output is null");
    OMNI_RETURN_VAL_IF(
        !backend::ValidateParallelLinear(call->type, backend::BackendKind::kOpenCL, args), false);
    const auto &first = args.projections[0];
    OMNI_RETURN_VAL_IF(!OpenCLOperator<backend::Linear>::Make(first, call), false);
    for (size_t index = 1U; index < args.projections.size(); ++index) {
      backend::OpenCLInvocation branch;
      branch.type = call->type;
      branch.has_nvidia_mma = call->has_nvidia_mma;
      OMNI_RETURN_VAL_IF(!OpenCLOperator<backend::Linear>::Make(args.projections[index], &branch),
                         false);
      OMNI_RETURN_VAL_IF_LOG(
          branch.entry != call->entry || branch.build_defines != call->build_defines ||
              branch.local_size != call->local_size,
          false, ERROR, "Parallel projections require identical kernel strategies");
      for (size_t dimension = 0U; dimension < 2U; ++dimension) {
        call->global_size[dimension] =
            std::max(call->global_size[dimension], branch.global_size[dimension]);
      }
    }
    call->dimensions = 3U;
    call->global_size[2] = args.projections.size();
    call->build_defines.emplace_back("OMNI_PARALLEL_LINEAR=1");
    for (size_t index = 1U; index < 3U; ++index) {
      const auto &projection = args.projections[std::min(index, args.projections.size() - 1U)];
      BindOpenCLArguments(call, projection.weight, projection.columns, projection.output);
    }
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
