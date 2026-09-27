#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace kv_cache_kernel {

template <typename T>
__device__ void StoreKvCacheImpl(const T *const __restrict__ key, const T *const __restrict__ value,
                                 const int32_t token_count, const int32_t key_value_size,
                                 const int32_t start_position, const int32_t cache_capacity,
                                 T *const __restrict__ key_cache,
                                 T *const __restrict__ value_cache) {
  const int64_t element_count = static_cast<int64_t>(token_count) * key_value_size;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; index < element_count; index += stride) {
    const int32_t token = static_cast<int32_t>(index / key_value_size);
    const int32_t cache_token = start_position + token;
    if (cache_token >= 0 && cache_token < cache_capacity) {
      const int32_t element = static_cast<int32_t>(index % key_value_size);
      const int64_t cache_index = static_cast<int64_t>(cache_token) * key_value_size + element;
      key_cache[cache_index] = key[index];
      value_cache[cache_index] = value[index];
    }
  }
}

} // namespace kv_cache_kernel
} // namespace kernels
} // namespace omni_runtime

