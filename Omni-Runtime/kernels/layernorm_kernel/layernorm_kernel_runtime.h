#pragma once

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class RmsNormKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const weight;
    int32_t rows;
    int32_t hidden_size;
    float epsilon;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.weight == nullptr || args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/layernorm_kernel/layernorm_kernel.cuh",
            "omni_runtime::kernels::layernorm_kernel::RmsNormImpl"),
        args.options, static_cast<const T *>(args.input), static_cast<const T *>(args.weight),
        args.rows, args.hidden_size, args.epsilon, static_cast<T *>(args.output));
  }
};

template <typename T> class LayernormKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const weight;
    const void *const bias;
    int32_t rows;
    int32_t hidden_size;
    float epsilon;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.weight == nullptr || args.bias == nullptr ||
                           args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/layernorm_kernel/layernorm_kernel.cuh",
            "omni_runtime::kernels::layernorm_kernel::LayernormImpl"),
        args.options, static_cast<const T *>(args.input), static_cast<const T *>(args.weight),
        static_cast<const T *>(args.bias), args.rows, args.hidden_size, args.epsilon,
        static_cast<T *>(args.output));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
