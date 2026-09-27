#pragma once

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime::kernels::flash_attention_kernel {

template <typename T, bool kHeadMajor>
__device__ void RunFlashAttention(const T *const query, const T *const key, const T *const value,
                                  const int32_t query_tokens, const int32_t query_heads,
                                  const int32_t kv_heads, const int32_t cached_tokens,
                                  const int32_t start_position, const int32_t head_dim,
                                  const bool is_causal, T *const output) {
  constexpr int32_t kQueries = 4;
  constexpr int32_t kKeys = 16;
  constexpr int32_t kMaxDimensions = 256;
  const int32_t dimensions = ((head_dim + 31) / 32) * 32;
  const int32_t key_stride = dimensions + 1;
  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  const int32_t token = blockIdx.x * kQueries + warp;
  const int32_t head = blockIdx.y;
  const int32_t kv_head = head / (query_heads / kv_heads);
  const int64_t query_offset = (kHeadMajor ? static_cast<int64_t>(head) * query_tokens + token
                                           : static_cast<int64_t>(token) * query_heads + head) *
                               head_dim;
  extern __shared__ __align__(16) unsigned char storage[];
  float *const queries = reinterpret_cast<float *>(storage);
  T *const keys = reinterpret_cast<T *>(queries + kQueries * dimensions);
  T *const values = keys + kKeys * key_stride;
  float *const probabilities = reinterpret_cast<float *>(values + kKeys * key_stride);
  for (int32_t d = lane; d < head_dim; d += 32) {
    queries[warp * dimensions + d] = token < query_tokens ? ToFloat(query[query_offset + d]) : 0.0F;
  }
  float accumulators[kMaxDimensions / 32] = {};
  float maximum = -INFINITY;
  float denominator = 0.0F;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int32_t limit =
      is_causal ? min(static_cast<int64_t>(cached_tokens),
                      static_cast<int64_t>(start_position) + (blockIdx.x + 1) * kQueries)
                : cached_tokens;
  for (int32_t first_key = 0; first_key < limit; first_key += kKeys) {
    for (int32_t index = threadIdx.x; index < kKeys * head_dim; index += blockDim.x) {
      const int32_t source = first_key + index / head_dim;
      const int32_t d = index % head_dim;
      const int64_t offset = (kHeadMajor ? static_cast<int64_t>(kv_head) * cached_tokens + source
                                         : static_cast<int64_t>(source) * kv_heads + kv_head) *
                                 head_dim +
                             d;
      keys[(index / head_dim) * key_stride + d] =
          source < cached_tokens ? key[offset] : FromFloat<T>(0.0F);
      values[(index / head_dim) * key_stride + d] =
          source < cached_tokens ? value[offset] : FromFloat<T>(0.0F);
    }
    __syncthreads();
    float score = 0.0F;
    for (int32_t d = lane % 2; d < head_dim; d += 2) {
      score += queries[warp * dimensions + d] * ToFloat(keys[(lane / 2) * key_stride + d]);
    }
    score += __shfl_xor_sync(0xFFFFFFFFU, score, 1);
    const int32_t source = first_key + lane / 2;
    score = token < query_tokens && source < cached_tokens &&
                    (!is_causal || source <= static_cast<int64_t>(start_position) + token)
                ? score * scale
                : -INFINITY;
    float next_maximum = fmaxf(maximum, score);
    for (int32_t offset = 16; offset > 0; offset /= 2) {
      next_maximum = fmaxf(next_maximum, __shfl_xor_sync(0xFFFFFFFFU, next_maximum, offset));
    }
    const float rescale = isfinite(next_maximum) ? expf(maximum - next_maximum) : 0.0F;
    const float probability = isfinite(next_maximum) ? expf(score - next_maximum) : 0.0F;
    if (lane % 2 == 0) {
      probabilities[warp * 32 + lane / 2] = probability;
    }
    float sum = lane % 2 == 0 ? probability : 0.0F;
    for (int32_t offset = 16; offset > 0; offset /= 2) {
      sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset);
    }
    denominator = denominator * rescale + sum;
    maximum = next_maximum;
    __syncwarp();
    for (int32_t item = 0; item < kMaxDimensions / 32; ++item) {
      const int32_t d = lane + item * 32;
      if (d < head_dim) {
        float weighted = 0.0F;
        for (int32_t index = 0; index < kKeys; ++index) {
          weighted += probabilities[warp * 32 + index] * ToFloat(values[index * key_stride + d]);
        }
        accumulators[item] = accumulators[item] * rescale + weighted;
      }
    }
    __syncthreads();
  }
  if (token < query_tokens) {
    for (int32_t item = 0; item < kMaxDimensions / 32; ++item) {
      const int32_t d = lane + item * 32;
      if (d < head_dim) {
        output[query_offset + d] = FromFloat<T>(accumulators[item] / denominator);
      }
    }
  }
}

template <typename T, bool kHeadMajor>
__global__ void FlashAttentionImpl(const T *const query, const T *const key, const T *const value,
                                   const int32_t query_tokens, const int32_t query_heads,
                                   const int32_t kv_heads, const int32_t cached_tokens,
                                   const int32_t start_position, const int32_t head_dim,
                                   const bool is_causal, T *const output) {
  RunFlashAttention<T, kHeadMajor>(query, key, value, query_tokens, query_heads, kv_heads,
                                   cached_tokens, start_position, head_dim, is_causal, output);
}

} // namespace omni_runtime::kernels::flash_attention_kernel
