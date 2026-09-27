#pragma once

#include "kernels/operator_runtime.h"

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

template <typename T> struct CudaOperator<T, backend::Embedding> final {
  static bool Launch(const backend::Embedding &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.indices.IsValid() || !args.weight.IsValid() ||
                               !args.output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.hidden_size <= 0, false, ERROR,
                           "Invalid compute operation dimensions");
    return EmbeddingKernelRuntime<T>::CompileAndLaunch(
        "backend_Embedding",
        {.options = {.grid_dim = {static_cast<uint32_t>(args.token_count), 1U, 1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .indices = CudaAddress<void>(args.indices),
         .weight = CudaAddress<void>(args.weight),
         .token_count = args.token_count,
         .hidden_size = args.hidden_size,
         .output = CudaAddress<void>(args.output)});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
