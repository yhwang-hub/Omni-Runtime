#pragma once

#include "kernels/operator_runtime_common.h"

#include "kernels/cuda_operator_runtime.h"
#include "Omni-Runtime/backend/backend_compute_define.h"

#include <cmath>

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class AdaRmsNormKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const scale;
    const void *const shift;
    int32_t rows;
    int32_t hidden_size;
    float epsilon;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.scale == nullptr || args.shift == nullptr ||
                           args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/adarms_kernel/adarms_kernel.cuh",
            "omni_runtime::kernels::adarms_kernel::AdaRmsNormImpl"),
        args.options, static_cast<const T *>(args.input), static_cast<const T *>(args.scale),
        static_cast<const T *>(args.shift), args.rows, args.hidden_size, args.epsilon,
        static_cast<T *>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::AdaRmsNorm> final {
  static bool Launch(const backend::AdaRmsNorm &args) {
    OMNI_RETURN_VAL_IF_LOG(
        args.rows <= 0 || args.hidden_size <= 0 || !std::isfinite(args.epsilon) ||
            args.epsilon <= 0.0F ||
            !HasElements(args.input, uint64_t(args.rows) * args.hidden_size, sizeof(T)) ||
            !HasElements(args.scale, args.hidden_size, sizeof(T)) ||
            !HasElements(args.shift, args.hidden_size, sizeof(T)) ||
            !HasElements(args.output, uint64_t(args.rows) * args.hidden_size, sizeof(T)),
        false, ERROR, "Invalid CUDA adaptive RMS normalization operation");
    return AdaRmsNormKernelRuntime<T>::CompileAndLaunch(
        "backend_AdaRmsNorm", {.options = {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
                                           .block_dim = {256U, 1U, 1U},
                                           .shared_memory_bytes = 0},
                               .input = CudaAddress<void>(args.input),
                               .scale = CudaAddress<void>(args.scale),
                               .shift = CudaAddress<void>(args.shift),
                               .rows = args.rows,
                               .hidden_size = args.hidden_size,
                               .epsilon = args.epsilon,
                               .output = CudaAddress<void>(args.output)});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
