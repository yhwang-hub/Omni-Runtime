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
  RunTensorAttention<T, kDimensions, kKeys, kStages, SynchronousStaging>(
      query, key, value, query_tokens, query_heads, kv_heads, cached_tokens, start_position,
      head_dim, is_causal, output);
}

} // namespace omni_runtime::kernels::flash_attention_v2_kernel
