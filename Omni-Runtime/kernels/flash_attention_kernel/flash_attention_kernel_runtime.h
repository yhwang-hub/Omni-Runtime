#pragma once

#include "kernels/operator_runtime.h"

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"
#include "kernels/gqa_attention_kernel/attention_define.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class FlashAttentionKernelRuntime final {
public:
  static constexpr int32_t kMaxHeadDim = 256;
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

  static int32_t SharedMemoryBytes(const int32_t head_dim) {
    const int32_t dimensions = ((head_dim + 31) / 32) * 32;
    return 4 * dimensions * sizeof(float) + 2 * 16 * (dimensions + 1) * sizeof(T) +
           4 * 32 * sizeof(float);
  }

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.query == nullptr || args.key == nullptr || args.value == nullptr ||
                           args.output == nullptr || args.heads <= 0 || args.tokens <= 0 ||
                           args.head_dim <= 0 || args.head_dim > 256,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T, true>(
            "kernels/flash_attention_kernel/flash_attention_kernel.cuh",
            "omni_runtime::kernels::flash_attention_kernel::FlashAttentionImpl"),
        {.grid_dim = {(static_cast<uint32_t>(args.tokens) + 3U) / 4U,
                      static_cast<uint32_t>(args.heads), 1U},
         .block_dim = {128U, 1U, 1U},
         .shared_memory_bytes = SharedMemoryBytes(args.head_dim)},
        static_cast<const T *>(args.query), static_cast<const T *>(args.key),
        static_cast<const T *>(args.value), args.tokens, args.heads, args.heads, args.tokens, 0,
        args.head_dim, args.is_causal, static_cast<T *>(args.output));
  }

  static bool CompileAndLaunch(const std::string &tag, const AttentionArgs &args) {
    OMNI_RETURN_VAL_IF(!IsValidAttention(args), false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T, false>(
            "kernels/flash_attention_kernel/flash_attention_kernel.cuh",
            "omni_runtime::kernels::flash_attention_kernel::FlashAttentionImpl"),
        {.grid_dim = {static_cast<uint32_t>((args.query_token_count + 3) / 4),
                      static_cast<uint32_t>(args.query_head_count), 1U},
         .block_dim = {128U, 1U, 1U},
         .shared_memory_bytes = SharedMemoryBytes(args.head_dim)},
        static_cast<const T *>(args.query), static_cast<const T *>(args.key_cache),
        static_cast<const T *>(args.value_cache), args.query_token_count, args.query_head_count,
        args.key_value_head_count, args.cached_token_count, args.start_position, args.head_dim,
        args.is_causal, static_cast<T *>(args.output));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
