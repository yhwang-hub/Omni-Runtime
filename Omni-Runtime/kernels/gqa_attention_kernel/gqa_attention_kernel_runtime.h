#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>

#include "kernels/kernel_runtime.h"
#include "kernels/gqa_attention_kernel/attention_define.h"
#include "kernels/flash_attention_v2_kernel/flash_attention_v2_kernel_runtime.h"
#include "kernels/flash_attention_kernel/flash_attention_kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class GqaAttentionKernelRuntime final {
public:
  using Args = AttentionArgs;

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(!IsValidAttention(args), false);
    if (args.query_token_count == 1 && args.decode_partials != nullptr) {
      const int32_t cached_token_count =
          args.is_causal
              ? static_cast<int32_t>(std::min<int64_t>(
                    args.cached_token_count, static_cast<int64_t>(args.start_position) + 1))
              : args.cached_token_count;
      const int32_t partitions = (cached_token_count + 127) / 128;
      const bool is_grouped =
          sizeof(T) == 2U && !std::is_integral_v<T> && CudaKernelRuntime::HasTensorMma() &&
          args.query_head_count / args.key_value_head_count == 4 && args.head_dim == 128 &&
          reinterpret_cast<uintptr_t>(args.query) % 4U == 0U &&
          reinterpret_cast<uintptr_t>(args.key_cache) % 16U == 0U &&
          reinterpret_cast<uintptr_t>(args.value_cache) % 16U == 0U;
      return CudaKernelRuntime::CompileAndLaunch(
                 tag + "_split",
                 CudaKernelRuntime::MakeKernelSource<T>(
                     "kernels/gqa_attention_kernel/decode_attention.cuh",
                     is_grouped
                         ? "omni_runtime::kernels::gqa_attention_kernel::GroupedDecodeAttentionImpl"
                         : "omni_runtime::kernels::gqa_attention_kernel::DecodeAttentionImpl"),
                 {.grid_dim = {static_cast<uint32_t>(is_grouped ? args.key_value_head_count
                                                                : args.query_head_count),
                               static_cast<uint32_t>(partitions), 1U},
                  .block_dim = {128U, 1U, 1U},
                  .shared_memory_bytes = 0},
                 static_cast<const T *>(args.query), static_cast<const T *>(args.key_cache),
                 static_cast<const T *>(args.value_cache), args.query_head_count,
                 args.key_value_head_count, cached_token_count, args.head_dim,
                 static_cast<float *>(args.decode_partials)) &&
             CudaKernelRuntime::CompileAndLaunch(
                 tag + "_merge",
                 CudaKernelRuntime::MakeKernelSource<T>(
                     "kernels/gqa_attention_kernel/decode_attention.cuh",
                     "omni_runtime::kernels::gqa_attention_kernel::MergeDecodeAttentionImpl"),
                 {.grid_dim = {static_cast<uint32_t>(args.query_head_count), 1U, 1U},
                  .block_dim = {256U, 1U, 1U},
                  .shared_memory_bytes = 0},
                 static_cast<const float *>(args.decode_partials), args.query_head_count,
                 partitions, args.head_dim, static_cast<T *>(args.output));
    }
    if constexpr (sizeof(T) == 2U && !std::is_integral_v<T>) {
      if (args.query_token_count > 1 && args.head_dim <= 128 && CudaKernelRuntime::HasTensorMma()) {
        return FlashAttentionV2KernelRuntime<T>::CompileAndLaunch(tag + "_tiled", args);
      }
    }
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/gqa_attention_kernel/gqa_attention_kernel.cuh",
            "omni_runtime::kernels::gqa_attention_kernel::GqaAttentionImpl"),
        {.grid_dim = {(static_cast<uint32_t>(args.query_token_count) + 3U) / 4U,
                      static_cast<uint32_t>(args.query_head_count), 1U},
         .block_dim = {128U, 1U, 1U},
         .shared_memory_bytes = FlashAttentionKernelRuntime<T>::SharedMemoryBytes(args.head_dim)},
        static_cast<const T *>(args.query), static_cast<const T *>(args.key_cache),
        static_cast<const T *>(args.value_cache), args.query_token_count, args.query_head_count,
        args.key_value_head_count, args.cached_token_count, args.start_position, args.head_dim,
        args.is_causal, static_cast<T *>(args.output));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
