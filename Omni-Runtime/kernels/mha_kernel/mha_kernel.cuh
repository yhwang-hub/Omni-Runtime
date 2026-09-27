#pragma once
#include <kernels/common/attention_kernel_impl.h>
namespace omni_runtime {
namespace kernels {
namespace mha_kernel {
__device__ void MhaFloatImpl(const float *const query, const float *const key,
                               const float *const value, const int32_t heads, const int32_t tokens,
                               const int32_t head_dim, const bool is_causal, float *const output) {
  attention_impl::RunOnlineAttention(query, key, value, heads, tokens, head_dim, is_causal, output);
}
} // namespace mha_kernel
} // namespace kernels
} // namespace omni_runtime
