#pragma once

#include <kernels/common/kernel_type_utils.h>
#include <kernels/matmul_kernel/tensor_core.cuh>

namespace omni_runtime::kernels::flash_attention_v2_kernel {

struct SynchronousStaging final {
  template <typename T, int32_t kDimensions, int32_t kKeys>
  static __device__ void Load(const T *const key, const T *const value, const int32_t first_key,
                              const int32_t cached_tokens, const int32_t kv_heads,
                              const int32_t kv_head, const int32_t head_dim, T *const keys,
                              T *const values) {
    constexpr int32_t kStride = kDimensions + 8;
    for (int32_t index = threadIdx.x * 8; index < kKeys * kDimensions; index += blockDim.x * 8) {
      const int32_t token = first_key + index / kDimensions;
      const int32_t d = index % kDimensions;
      const bool is_valid = token < cached_tokens && d < head_dim;
      const int64_t offset =
          is_valid ? (static_cast<int64_t>(token) * kv_heads + kv_head) * head_dim + d : 0;
      const int32_t destination = (index / kDimensions) * kStride + d;
#pragma unroll
      for (int32_t item = 0; item < 8; ++item) {
        const bool has_value = is_valid && d + item < head_dim;
        keys[destination + item] = has_value ? key[offset + item] : FromFloat<T>(0.0F);
        values[destination + item] = has_value ? value[offset + item] : FromFloat<T>(0.0F);
      }
    }
  }

  static __device__ void Wait() {
    __syncthreads();
  }
};

template <typename T, int32_t kDimensions, int32_t kKeys, int32_t kStages, typename Staging>
__device__ void RunTensorAttention(const T *const query, const T *const key, const T *const value,
                                   const int32_t query_tokens, const int32_t query_heads,
                                   const int32_t kv_heads, const int32_t cached_tokens,
                                   const int32_t start_position, const int32_t head_dim,
                                   const bool is_causal, T *const output) {
  constexpr int32_t kQueries = kDimensions <= 80 ? 128 : 64;
  constexpr int32_t kStride = kDimensions + 8;
  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  const int32_t row = lane / 4;
  const int32_t column = (lane % 4) * 2;
  const int32_t head = blockIdx.y;
  const int32_t kv_head = head / (query_heads / kv_heads);
  const int32_t first_query = blockIdx.x * kQueries + warp * 16;
  constexpr bool kDynamic = 2 * kStages * kKeys * kStride * sizeof(T) > 48U * 1024U;
  constexpr int32_t kStaticElements = kDynamic ? 1 : kStages * kKeys * kStride;
  __shared__ T static_keys[kStaticElements];
  __shared__ T static_values[kStaticElements];
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  T *const staged_keys = kDynamic ? reinterpret_cast<T *>(dynamic_storage) : static_keys;
  T *const staged_values = kDynamic ? staged_keys + kStages * kKeys * kStride : static_values;
  // PTX fragment layout keeps Q, scores, softmax and rescaled O in registers.
  uint32_t queries[kDimensions / 16][4];
  float accumulators[kDimensions / 8][4] = {};
  float maxima[2] = {-INFINITY, -INFINITY};
  float denominators[2] = {};
#pragma unroll
  for (int32_t dimension = 0; dimension < kDimensions; dimension += 16) {
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t token = first_query + row + (item % 2) * 8;
      const int32_t d = dimension + column + (item / 2) * 8;
      const int64_t offset = (static_cast<int64_t>(token) * query_heads + head) * head_dim + d;
      queries[dimension / 16][item] =
          token < query_tokens && d < head_dim
              ? PackPair(query[offset], d + 1 < head_dim ? query[offset + 1] : FromFloat<T>(0.0F))
              : 0U;
    }
  }
  const int32_t limit = is_causal ? min(static_cast<int64_t>(cached_tokens),
                                        static_cast<int64_t>(start_position) +
                                            static_cast<int64_t>(blockIdx.x + 1) * kQueries)
                                  : cached_tokens;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  if constexpr (kStages == 2) {
    Staging::template Load<T, kDimensions, kKeys>(key, value, 0, cached_tokens, kv_heads, kv_head,
                                                  head_dim, staged_keys, staged_values);
  }
  for (int32_t first_key = 0; first_key < limit; first_key += kKeys) {
    const int32_t stage = (first_key / kKeys) % kStages;
    T *const keys = staged_keys + stage * kKeys * kStride;
    T *const values = staged_values + stage * kKeys * kStride;
    if constexpr (kStages == 1) {
      Staging::template Load<T, kDimensions, kKeys>(key, value, first_key, cached_tokens, kv_heads,
                                                    kv_head, head_dim, keys, values);
    }
    Staging::Wait();
    if constexpr (kStages == 2) {
      if (first_key + kKeys < limit) {
        const int32_t next_stage = 1 - stage;
        Staging::template Load<T, kDimensions, kKeys>(key, value, first_key + kKeys, cached_tokens,
                                                      kv_heads, kv_head, head_dim,
                                                      staged_keys + next_stage * kKeys * kStride,
                                                      staged_values + next_stage * kKeys * kStride);
      }
    }
    float scores[kKeys / 8][4] = {};
#pragma unroll
    for (int32_t d = 0; d < kDimensions; d += 16) {
#pragma unroll
      for (int32_t tile = 0; tile < kKeys / 8; ++tile) {
        const int32_t offset = (tile * 8 + row) * kStride + d + column;
        const uint32_t b[2] = {*reinterpret_cast<const uint32_t *>(keys + offset),
                               *reinterpret_cast<const uint32_t *>(keys + offset + 8)};
        TensorMma<T>(queries[d / 16], b, scores[tile]);
      }
    }
#pragma unroll
    for (int32_t item_row = 0; item_row < 2; ++item_row) {
      const int32_t token = first_query + row + item_row * 8;
      float maximum = maxima[item_row];
#pragma unroll
      for (int32_t tile = 0; tile < kKeys / 8; ++tile) {
#pragma unroll
        for (int32_t item = 0; item < 2; ++item) {
          const int32_t source = first_key + tile * 8 + column + item;
          float &score = scores[tile][item_row * 2 + item];
          score = token < query_tokens && source < cached_tokens &&
                          (!is_causal || source <= start_position + token)
                      ? score * scale
                      : -INFINITY;
          maximum = fmaxf(maximum, score);
        }
      }
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, 1));
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, 2));
      const float rescale = isfinite(maximum) ? expf(maxima[item_row] - maximum) : 0.0F;
      float sum = 0.0F;
#pragma unroll
      for (int32_t tile = 0; tile < kKeys / 8; ++tile) {
#pragma unroll
        for (int32_t item = 0; item < 2; ++item) {
          const float probability =
              isfinite(maximum) ? expf(scores[tile][item_row * 2 + item] - maximum) : 0.0F;
          scores[tile][item_row * 2 + item] = probability;
          sum += probability;
        }
      }
      sum += __shfl_xor_sync(0xffffffffU, sum, 1);
      sum += __shfl_xor_sync(0xffffffffU, sum, 2);
      denominators[item_row] = denominators[item_row] * rescale + sum;
      maxima[item_row] = maximum;
#pragma unroll
      for (int32_t tile = 0; tile < kDimensions / 8; ++tile) {
        accumulators[tile][item_row * 2] *= rescale;
        accumulators[tile][item_row * 2 + 1] *= rescale;
      }
    }
#pragma unroll
    for (int32_t source = 0; source < kKeys; source += 16) {
      uint32_t a[4];
#pragma unroll
      for (int32_t item = 0; item < 4; ++item) {
        const int32_t tile = source / 8 + item / 2;
        const int32_t offset = (item % 2) * 2;
        a[item] =
            PackPair(FromFloat<T>(scores[tile][offset]), FromFloat<T>(scores[tile][offset + 1]));
      }
#pragma unroll
      for (int32_t tile = 0; tile < kDimensions / 8; ++tile) {
        const int32_t d = tile * 8 + row;
        const uint32_t b[2] = {PackPair(values[(source + column) * kStride + d],
                                        values[(source + column + 1) * kStride + d]),
                               PackPair(values[(source + column + 8) * kStride + d],
                                        values[(source + column + 9) * kStride + d])};
        TensorMma<T>(a, b, accumulators[tile]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int32_t tile = 0; tile < kDimensions / 8; ++tile) {
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t token = first_query + row + (item / 2) * 8;
      const int32_t d = tile * 8 + column + item % 2;
      if (token < query_tokens && d < head_dim) {
        output[(static_cast<int64_t>(token) * query_heads + head) * head_dim + d] =
            FromFloat<T>(accumulators[tile][item] / denominators[item / 2]);
      }
    }
  }
}

} // namespace omni_runtime::kernels::flash_attention_v2_kernel
