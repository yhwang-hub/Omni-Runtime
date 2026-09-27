#pragma once

#include "kernels/operator_runtime.h"
#include "kernels/flash_attention_v4_kernel/flash_attention_v4_kernel_runtime.h"

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
          args.is_causal && args.cache_position == nullptr
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
                 static_cast<float *>(args.decode_partials), args.cache_position) &&
             CudaKernelRuntime::CompileAndLaunch(
                 tag + "_merge",
                 CudaKernelRuntime::MakeKernelSource<T>(
                     "kernels/gqa_attention_kernel/decode_attention.cuh",
                     "omni_runtime::kernels::gqa_attention_kernel::MergeDecodeAttentionImpl"),
                 {.grid_dim = {static_cast<uint32_t>(args.query_head_count), 1U, 1U},
                  .block_dim = {256U, 1U, 1U},
                  .shared_memory_bytes = 0},
                 static_cast<const float *>(args.decode_partials), args.query_head_count,
                 partitions, args.head_dim, static_cast<T *>(args.output), args.cache_position);
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

template <typename T> struct CudaOperator<T, backend::Attention> final {
  static bool Launch(const backend::Attention &args) {
    OMNI_RETURN_VAL_IF_LOG(
        !args.query.IsValid() || !args.key_cache.IsValid() || !args.value_cache.IsValid() ||
            !args.output.IsValid() ||
            ((!args.decode_partials.IsValid() ||
              args.decode_partials.backend_kind != backend::BackendKind::kCuda) &&
             (args.decode_partials.data != nullptr || args.decode_partials.offset_bytes != 0U ||
              args.decode_partials.size_bytes != 0U)) ||
            ((!args.cache_position.IsValid() ||
              args.cache_position.backend_kind != backend::BackendKind::kCuda) &&
             (args.cache_position.data != nullptr || args.cache_position.offset_bytes != 0U ||
              args.cache_position.size_bytes != 0U)),
        false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.query_token_count <= 0 || args.query_head_count <= 0 ||
                               args.key_value_head_count <= 0 || args.cached_token_count <= 0 ||
                               args.head_dim <= 0,
                           false, ERROR, "Invalid compute operation dimensions");
    const AttentionArgs parameters = {
        .options = {.grid_dim = {static_cast<uint32_t>(uint64_t(args.query_token_count) *
                                                       args.query_head_count),
                                 1U, 1U},
                    .block_dim = {128U, 1U, 1U},
                    .shared_memory_bytes = 0},
        .query = CudaAddress<void>(args.query),
        .key_cache = CudaAddress<void>(args.key_cache),
        .value_cache = CudaAddress<void>(args.value_cache),
        .query_token_count = args.query_token_count,
        .query_head_count = args.query_head_count,
        .key_value_head_count = args.key_value_head_count,
        .cached_token_count = args.cached_token_count,
        .start_position = args.start_position,
        .head_dim = args.head_dim,
        .is_causal = args.is_causal,
        .output = CudaAddress<void>(args.output),
        .decode_partials = CudaAddress<void>(args.decode_partials),
        .cache_position = CudaAddress<const int32_t>(args.cache_position)};
    if constexpr (sizeof(T) == 2U) {
      if (args.query_token_count > 1 && args.head_dim <= 128 &&
          kernel::jit::JitRuntime::HasSm120()) {
        return FlashAttentionV4KernelRuntime<T>::CompileAndLaunch("backend_Attention_v4_sm120",
                                                                  parameters);
      }
    }
    return GqaAttentionKernelRuntime<T>::CompileAndLaunch("backend_Attention", parameters);
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
