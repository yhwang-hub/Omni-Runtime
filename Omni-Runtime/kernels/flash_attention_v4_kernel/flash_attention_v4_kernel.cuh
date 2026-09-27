#pragma once

#include <kernels/flash_attention_v2_kernel/attention_impl.cuh>

namespace omni_runtime::kernels::flash_attention_v4_kernel {

struct AsynchronousStaging final {
  template <typename T, int32_t kDimensions, int32_t kKeys>
  static __device__ void Load(const T *const key, const T *const value, const int32_t first_key,
                              const int32_t cached_tokens, const int32_t kv_heads,
                              const int32_t kv_head, const int32_t head_dim, T *const keys,
                              T *const values) {
    constexpr int32_t kStride = kDimensions + 8;
    for (int32_t index = threadIdx.x * 8; index < kKeys * kDimensions; index += blockDim.x * 8) {
      const int32_t token = first_key + index / kDimensions;
      const int32_t d = index % kDimensions;
      const int32_t destination = (index / kDimensions) * kStride + d;
      const bool is_valid = token < cached_tokens && d < head_dim;
      const int64_t offset =
          is_valid ? (static_cast<int64_t>(token) * kv_heads + kv_head) * head_dim + d : 0;
      if (!is_valid || (d + 7 < head_dim && reinterpret_cast<uintptr_t>(key + offset) % 16U == 0U &&
                        reinterpret_cast<uintptr_t>(value + offset) % 16U == 0U)) {
        const uint32_t key_address =
            static_cast<uint32_t>(__cvta_generic_to_shared(keys + destination));
        const uint32_t value_address =
            static_cast<uint32_t>(__cvta_generic_to_shared(values + destination));
        const int32_t bytes = is_valid ? 16 : 0;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(key_address),
                     "l"(key + offset), "r"(bytes)
                     : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(value_address),
                     "l"(value + offset), "r"(bytes)
                     : "memory");
        continue;
      }
#pragma unroll
      for (int32_t item = 0; item < 8; ++item) {
        const bool has_value = is_valid && d + item < head_dim;
        keys[destination + item] = has_value ? key[offset + item] : FromFloat<T>(0.0F);
        values[destination + item] = has_value ? value[offset + item] : FromFloat<T>(0.0F);
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }

  static __device__ void Wait() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
  }
};

template <typename T, int32_t kDimensions, int32_t kKeys, int32_t kStages>
__global__ void BlackwellAttentionImpl(const T *const query, const T *const key,
                                       const T *const value, const int32_t query_tokens,
                                       const int32_t query_heads, const int32_t kv_heads,
                                       const int32_t cached_tokens, const int32_t start_position,
                                       const int32_t head_dim, const bool is_causal,
                                       T *const output) {
  flash_attention_v2_kernel::RunTensorAttention<T, kDimensions, kKeys, kStages,
                                                AsynchronousStaging>(
      query, key, value, query_tokens, query_heads, kv_heads, cached_tokens, start_position,
      head_dim, is_causal, output);
}

} // namespace omni_runtime::kernels::flash_attention_v4_kernel
