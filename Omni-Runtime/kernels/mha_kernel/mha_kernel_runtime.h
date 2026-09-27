#pragma once

#include "kernels/operator_runtime.h"
#include "Omni-Runtime/backend/backend_compute_define.h"

#include <cstdint>
#include <string>

#include "kernels/flash_attention_kernel/flash_attention_kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class MhaKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const query;
    const void *const key;
    const void *const value;
    int32_t heads;
    int32_t tokens;
    int32_t head_dim;
    bool is_causal;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.query == nullptr || args.key == nullptr || args.value == nullptr ||
                           args.output == nullptr || args.heads <= 0 || args.tokens <= 0 ||
                           args.head_dim <= 0 || args.head_dim > 256,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>("kernels/mha_kernel/mha_kernel.cuh",
                                               "omni_runtime::kernels::mha_kernel::MhaImpl"),
        {.grid_dim = {(static_cast<uint32_t>(args.tokens) + 3U) / 4U,
                      static_cast<uint32_t>(args.heads), 1U},
         .block_dim = {128U, 1U, 1U},
         .shared_memory_bytes = FlashAttentionKernelRuntime<T>::SharedMemoryBytes(args.head_dim)},
        static_cast<const T *>(args.query), static_cast<const T *>(args.key),
        static_cast<const T *>(args.value), args.heads, args.tokens, args.head_dim, args.is_causal,
        static_cast<T *>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::MultiHeadAttention> final {
  static bool Launch(const backend::MultiHeadAttention &args) {
    OMNI_RETURN_VAL_IF_LOG(args.heads <= 0 || args.heads > 65535 || args.tokens <= 0 ||
                               args.head_dim <= 0 || args.head_dim > 256,
                           false, ERROR, "Invalid CUDA standalone MHA dimensions");
    const uint64_t count = uint64_t(args.heads) * args.tokens * args.head_dim;
    OMNI_RETURN_VAL_IF_LOG(!HasElements(args.query, count, sizeof(T)) ||
                               !HasElements(args.key, count, sizeof(T)) ||
                               !HasElements(args.value, count, sizeof(T)) ||
                               !HasElements(args.output, count, sizeof(T)),
                           false, ERROR, "Invalid CUDA standalone MHA buffers");
    return MhaKernelRuntime<T>::CompileAndLaunch("backend_MultiHeadAttention",
                                                 {.options = {},
                                                  .query = CudaAddress<void>(args.query),
                                                  .key = CudaAddress<void>(args.key),
                                                  .value = CudaAddress<void>(args.value),
                                                  .heads = args.heads,
                                                  .tokens = args.tokens,
                                                  .head_dim = args.head_dim,
                                                  .is_causal = args.is_causal,
                                                  .output = CudaAddress<void>(args.output)});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
