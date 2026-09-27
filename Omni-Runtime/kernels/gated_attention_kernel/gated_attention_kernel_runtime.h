#pragma once

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class SplitGatedQueryKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const query_gate;
    int32_t token_count;
    int32_t head_count;
    int32_t head_size;
    void *const query;
    void *const gate;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.query_gate == nullptr || args.query == nullptr || args.gate == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/gated_attention_kernel/gated_attention_kernel.cuh",
            "omni_runtime::kernels::gated_attention_kernel::SplitGatedQueryImpl"),
        args.options, static_cast<const T *>(args.query_gate), args.token_count, args.head_count,
        args.head_size, static_cast<T *>(args.query), static_cast<T *>(args.gate));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
