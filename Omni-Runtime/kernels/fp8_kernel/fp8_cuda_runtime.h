#pragma once

#include "Omni-Runtime/backend/backend_cuda_utils.h"
#include "kernels/fp8_kernel/fp8_runtime_utils.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <typename T> struct CudaOperator<T, backend::Fp8Linear> final {
  static bool Launch(const backend::Fp8Linear &args, uint8_t *const activation,
                     float *const activation_scales) {
    const auto *const input = backend::cuda_internal::CudaAddress<const T>(args.input);
    const auto *const weight = backend::cuda_internal::CudaAddress<const uint8_t>(args.weight);
    const auto *const scales = backend::cuda_internal::CudaAddress<const float>(args.weight_scales);
    auto *const output = backend::cuda_internal::CudaAddress<T>(args.output);
    OMNI_RETURN_VAL_IF_LOG(input == nullptr || weight == nullptr || scales == nullptr ||
                               output == nullptr || activation == nullptr ||
                               activation_scales == nullptr,
                           false, ERROR, "FP8 device pointer is null");
    const bool is_decode = args.rows <= 8;
    const int32_t tile_columns = is_decode ? 1 : 32;
    const int32_t tile_rows = args.rows <= 16 ? 16 : 32;
    OMNI_RETURN_VAL_IF_LOG(!is_decode && !CudaKernelRuntime::HasTensorMma(), false, ERROR,
                           "FP8 prefill requires tensor MMA support");
    const std::string include = "kernels/fp8_kernel/fp8_kernel.cuh";
    const std::string quantize = "omni_runtime::kernels::fp8_kernel::QuantizeImpl";
    OMNI_RETURN_VAL_IF(!CudaKernelRuntime::CompileAndLaunch(
                           "fp8_quantize",
                           is_decode ? CudaKernelRuntime::MakeKernelSource<T, 1>(include, quantize)
                                     : CudaKernelRuntime::MakeKernelSource<T, 0>(include, quantize),
                           {.grid_dim = {static_cast<uint32_t>((args.inner / 128 + 7) / 8),
                                         static_cast<uint32_t>(args.rows), 1U},
                            .block_dim = {256U, 1U, 1U},
                            .shared_memory_bytes = 0},
                           input, args.inner, activation, activation_scales),
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        is_decode ? "fp8_gemv" : "fp8_matmul", MakeSource(args, is_decode),
        {.grid_dim = {static_cast<uint32_t>((args.columns + tile_columns - 1) / tile_columns),
                      static_cast<uint32_t>(is_decode ? args.rows
                                                      : (args.rows + tile_rows - 1) / tile_rows),
                      1U},
         .block_dim = {128U, 1U, 1U},
         .shared_memory_bytes = 0},
        activation, activation_scales, weight, scales, args.rows, args.inner, args.columns,
        args.block_reduction_groups, output);
  }

private:
  static std::string MakeSource(const backend::Fp8Linear &args, const bool is_decode) {
    return reinterpret_cast<uintptr_t>(
               backend::cuda_internal::CudaAddress<const uint8_t>(args.weight)) %
                       16U ==
                   0U
               ? MakeAlignedSource<1>(args, is_decode)
               : MakeAlignedSource<0>(args, is_decode);
  }

  template <int32_t kIsAligned>
  static std::string MakeAlignedSource(const backend::Fp8Linear &args, const bool is_decode) {
    const std::string include = "kernels/fp8_kernel/fp8_kernel.cuh";
    const std::string kernel = is_decode ? "omni_runtime::kernels::fp8_kernel::GemvImpl"
                                         : "omni_runtime::kernels::fp8_kernel::MatmulImpl";
    if (is_decode) {
      return CudaKernelRuntime::MakeKernelSource<T, kIsAligned>(include, kernel);
    }
    return args.rows <= 16
               ? CudaKernelRuntime::MakeKernelSource<T, kIsAligned, 16>(include, kernel)
               : CudaKernelRuntime::MakeKernelSource<T, kIsAligned, 32>(include, kernel);
  }
};

} // namespace omni_runtime::kernels::runtime
