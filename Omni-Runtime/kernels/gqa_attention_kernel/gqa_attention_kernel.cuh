#pragma once
#include <cstdint>

#include <kernels/flash_attention_kernel/flash_attention_kernel.cuh>

namespace omni_runtime {
namespace kernels {
namespace gqa_attention_kernel {

template <typename T>
__global__ void
GqaAttentionImpl(const T *const __restrict__ query, const T *const __restrict__ key_cache,
                 const T *const __restrict__ value_cache, const int32_t query_token_count,
                 const int32_t query_head_count, const int32_t key_value_head_count,
                 const int32_t cached_token_count, const int32_t start_position,
                 const int32_t head_dim, const bool is_causal, T *const __restrict__ output) {
  flash_attention_kernel::RunFlashAttention<T, false>(
      query, key_cache, value_cache, query_token_count, query_head_count, key_value_head_count,
      cached_token_count, start_position, head_dim, is_causal, output);
}

} // namespace gqa_attention_kernel
} // namespace kernels
} // namespace omni_runtime
