#pragma once

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class EmbeddingKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const indices;
    const void *const weight;
    int32_t token_count;
    int32_t hidden_size;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.indices == nullptr || args.weight == nullptr || args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/embedding_kernel/embedding_kernel.cuh",
            "omni_runtime::kernels::embedding_kernel::EmbeddingImpl"),
        args.options, static_cast<const T *>(args.indices), static_cast<const T *>(args.weight),
        args.token_count, args.hidden_size, static_cast<T *>(args.output));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
