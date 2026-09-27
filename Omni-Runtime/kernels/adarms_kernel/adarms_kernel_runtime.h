#pragma once

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

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
