#pragma once
#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace kv_cache_kernel {

template <typename T>
__global__ void StoreKvCacheImpl(const T *const __restrict__ key, const T *const __restrict__ value,
                                 const int32_t token_count, const int32_t key_value_size,
                                 const int32_t start_position, const int32_t cache_capacity,
                                 T *const __restrict__ key_cache, T *const __restrict__ value_cache,
                                 const int32_t *const cache_position, const bool is_fp16_cache,
                                 const bool is_fp16_key_value) {
  const int64_t element_count = static_cast<int64_t>(token_count) * key_value_size;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; index < element_count; index += stride) {
    const int32_t token = static_cast<int32_t>(index / key_value_size);
    const int32_t cache_token =
        (cache_position == nullptr ? start_position : *cache_position) + token;
    if (cache_token >= 0 && cache_token < cache_capacity) {
      const int32_t element = static_cast<int32_t>(index % key_value_size);
      const int64_t cache_index = static_cast<int64_t>(cache_token) * key_value_size + element;
      if (is_fp16_key_value) {
        reinterpret_cast<__half *>(key_cache)[cache_index] = __float2half_rn(ToFloat(key[index]));
        reinterpret_cast<__half *>(value_cache)[cache_index] =
            __float2half_rn(ToFloat(value[index]));
        continue;
      }
      key_cache[cache_index] =
          is_fp16_cache ? FromFloat<T>(__half2float(__float2half_rn(ToFloat(key[index]))))
                        : key[index];
      value_cache[cache_index] =
          is_fp16_cache ? FromFloat<T>(__half2float(__float2half_rn(ToFloat(value[index]))))
                        : value[index];
    }
  }
}

template <typename T>
__global__ void AdvanceSequenceImpl(int32_t *const positions, int32_t *const cache_position) {
  if (threadIdx.x == 0) {
    ++positions[0];
    ++positions[1];
    ++positions[2];
    ++*cache_position;
  }
}

} // namespace kv_cache_kernel
} // namespace kernels
} // namespace omni_runtime
