#pragma once
#include <kernels/flash_attention_kernel/flash_attention_kernel.cuh>
namespace omni_runtime {
namespace kernels {
namespace mha_kernel {
template <typename T>
__global__ void MhaImpl(const T *const query, const T *const key, const T *const value,
                        const int32_t heads, const int32_t tokens, const int32_t head_dim,
                        const bool is_causal, T *const output) {
  flash_attention_kernel::RunFlashAttention<T, true>(query, key, value, tokens, heads, heads,
                                                     tokens, 0, head_dim, is_causal, output);
}
} // namespace mha_kernel
} // namespace kernels
} // namespace omni_runtime
