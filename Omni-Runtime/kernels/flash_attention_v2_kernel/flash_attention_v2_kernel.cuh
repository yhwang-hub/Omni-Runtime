#pragma once

#include <kernels/flash_attention_v2_kernel/attention_impl.cuh>

namespace omni_runtime::kernels::flash_attention_v2_kernel {

template <typename T, int32_t kDimensions, int32_t kKeys, int32_t kStages>
__global__ void MmaAttentionImpl(const T *const query, const T *const key, const T *const value,
                                 const int32_t query_tokens, const int32_t query_heads,
                                 const int32_t kv_heads, const int32_t cached_tokens,
                                 const int32_t start_position, const int32_t head_dim,
                                 const bool is_causal, T *const output) {
  static_assert(kStages == 1);
  RunTensorAttention<T, kDimensions, kKeys, kStages, SynchronousStaging, 0>(
      query, key, value, query_tokens, query_heads, kv_heads, cached_tokens, start_position,
      head_dim, is_causal, 0, output);
}

template <typename T, int32_t kDimensions, int32_t kKeys, int32_t kIsGgml, int32_t kIsFp16KeyValue>
__global__ void HalfAttentionImpl(const T *const query, const T *const key, const T *const value,
                                  const int32_t query_tokens, const int32_t query_heads,
                                  const int32_t kv_heads, const int32_t cached_tokens,
                                  const int32_t start_position, const int32_t head_dim,
                                  const bool is_causal, const int32_t stream_groups,
                                  const int32_t *const cache_position, T *const output) {
  const int32_t position =
      kIsGgml != 0 && cache_position != nullptr ? *cache_position : start_position;
  const int32_t count = kIsGgml != 0 && cache_position != nullptr ? position + 1 : cached_tokens;
  using Cache = std::conditional_t<kIsFp16KeyValue != 0, __half, T>;
  RunTensorAttention<__half, kDimensions, kKeys, 1, SynchronousStaging, kIsGgml>(
      query, reinterpret_cast<const Cache *>(key), reinterpret_cast<const Cache *>(value),
      query_tokens, query_heads, kv_heads, count, position, head_dim, is_causal, stream_groups,
      output);
}

} // namespace omni_runtime::kernels::flash_attention_v2_kernel
