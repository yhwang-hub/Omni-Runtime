#pragma once

#include <kernels/common/kernel_type_utils.h>
#include <kernels/matmul_kernel/tensor_core.cuh>

namespace omni_runtime {
namespace kernels {
namespace gqa_attention_kernel {

template <typename T>
__global__ void GroupedDecodeAttentionImpl(const T *const query, const T *const key,
                                           const T *const value, const int32_t query_heads,
                                           const int32_t kv_heads, const int32_t cached_tokens,
                                           const int32_t head_dim, float *const partials) {
  constexpr int32_t kPartition = 128;
  constexpr int32_t kStride = 136;
  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  const int32_t row = lane / 4;
  const int32_t column = (lane % 4) * 2;
  const int32_t kv_head = blockIdx.x;
  const int32_t first_token = blockIdx.y * kPartition;
  const int32_t count = min(kPartition, cached_tokens - first_token);
  __shared__ __align__(16) T cache[kPartition * kStride];
  __shared__ float scores[4][kPartition];
  constexpr int32_t kValues = 16 / sizeof(T);
  for (int32_t index = threadIdx.x * kValues; index < kPartition * 128;
       index += blockDim.x * kValues) {
    const int32_t token = index / 128;
    const int32_t d = index % 128;
    *reinterpret_cast<uint4 *>(cache + token * kStride + d) =
        token < count
            ? *reinterpret_cast<const uint4 *>(
                  key + (static_cast<int64_t>(first_token + token) * kv_heads + kv_head) * 128 + d)
            : make_uint4(0U, 0U, 0U, 0U);
  }
  __syncthreads();
  float fragments[4][4] = {};
#pragma unroll
  for (int32_t d = 0; d < 128; d += 16) {
    const int32_t query_offset = (kv_head * 4 + row) * 128 + d + column;
    const uint32_t a[4] = {
        row < 4 ? *reinterpret_cast<const uint32_t *>(query + query_offset) : 0U, 0U,
        row < 4 ? *reinterpret_cast<const uint32_t *>(query + query_offset + 8) : 0U, 0U};
#pragma unroll
    for (int32_t tile = 0; tile < 4; ++tile) {
      const int32_t offset = (warp * 32 + tile * 8 + row) * kStride + d + column;
      const uint32_t b[2] = {*reinterpret_cast<const uint32_t *>(cache + offset),
                             *reinterpret_cast<const uint32_t *>(cache + offset + 8)};
      TensorMma<T>(a, b, fragments[tile]);
    }
  }
  if (row < 4) {
#pragma unroll
    for (int32_t tile = 0; tile < 4; ++tile) {
#pragma unroll
      for (int32_t item = 0; item < 2; ++item) {
        const int32_t token = warp * 32 + tile * 8 + column + item;
        scores[row][token] = token < count ? fragments[tile][item] * rsqrtf(128.0F) : -INFINITY;
      }
    }
  }
  __syncthreads();
  float maximum = -INFINITY;
#pragma unroll
  for (int32_t token = lane; token < kPartition; token += 32) {
    maximum = fmaxf(maximum, scores[warp][token]);
  }
  maximum = WarpReduceMax(maximum);
  maximum = __shfl_sync(0xffffffffU, maximum, 0);
  float denominator = 0.0F;
#pragma unroll
  for (int32_t token = lane; token < kPartition; token += 32) {
    const float probability = expf(scores[warp][token] - maximum);
    scores[warp][token] = probability;
    denominator += probability;
  }
  denominator = WarpReduceSum(denominator);
  for (int32_t index = threadIdx.x * kValues; index < kPartition * 128;
       index += blockDim.x * kValues) {
    const int32_t token = index / 128;
    const int32_t d = index % 128;
    if (token < count) {
      *reinterpret_cast<uint4 *>(cache + token * kStride + d) = *reinterpret_cast<const uint4 *>(
          value + (static_cast<int64_t>(first_token + token) * kv_heads + kv_head) * 128 + d);
    }
  }
  __syncthreads();
  const int32_t head = kv_head * 4 + warp;
  const int64_t output_offset =
      (static_cast<int64_t>(blockIdx.y) * query_heads + head) * (head_dim + 2);
  float accumulators[4] = {};
  for (int32_t token = 0; token < count; ++token) {
    const float probability = scores[warp][token];
    const uint2 packed = *reinterpret_cast<const uint2 *>(cache + token * kStride + lane * 4);
    const T *const values = reinterpret_cast<const T *>(&packed);
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      accumulators[item] = fmaf(probability, ToFloat(values[item]), accumulators[item]);
    }
  }
#pragma unroll
  for (int32_t item = 0; item < 4; ++item) {
    partials[output_offset + lane * 4 + item] = accumulators[item];
  }
  if (lane == 0) {
    partials[output_offset + head_dim] = maximum;
    partials[output_offset + head_dim + 1] = denominator;
  }
}

template <typename T>
__global__ void DecodeAttentionImpl(const T *const query, const T *const key, const T *const value,
                                    const int32_t query_heads, const int32_t kv_heads,
                                    const int32_t cached_tokens, const int32_t head_dim,
                                    float *const partials) {
  constexpr int32_t kPartition = 128;
  const int32_t head = blockIdx.x;
  const int32_t partition = blockIdx.y;
  const int32_t lane = threadIdx.x % 8;
  const int32_t kv_head = head / (query_heads / kv_heads);
  __shared__ float scores[kPartition];
  __shared__ float scratch[4];
  const int32_t first_token = partition * kPartition;
  const int32_t limit = min(cached_tokens, (partition + 1) * kPartition);
  for (int32_t index = threadIdx.x / 8; index < kPartition; index += blockDim.x / 8) {
    const int32_t token = first_token + index;
    float score = 0.0F;
    const int64_t offset = (static_cast<int64_t>(token) * kv_heads + kv_head) * head_dim;
    for (int32_t dimension = lane; dimension < head_dim; dimension += 8) {
      if (token < limit) {
        score = fmaf(ToFloat(query[head * head_dim + dimension]), ToFloat(key[offset + dimension]),
                     score);
      }
    }
    for (int32_t delta = 4; delta > 0; delta /= 2) {
      score += __shfl_xor_sync(0xffffffffU, score, delta);
    }
    if (lane == 0) {
      scores[index] = token < limit ? score * rsqrtf(static_cast<float>(head_dim)) : -INFINITY;
    }
  }
  __syncthreads();
  const float maximum = BlockReduceMax(scores[threadIdx.x], scratch);
  const float probability = expf(scores[threadIdx.x] - maximum);
  const float denominator = BlockReduceSum(probability, scratch);
  scores[threadIdx.x] = probability;
  __syncthreads();
  const int64_t output_offset =
      (static_cast<int64_t>(partition) * query_heads + head) * (head_dim + 2);
  for (int32_t dimension = threadIdx.x; dimension < head_dim; dimension += blockDim.x) {
    float accumulator = 0.0F;
    for (int32_t token = first_token; token < limit; ++token) {
      const int64_t offset = (static_cast<int64_t>(token) * kv_heads + kv_head) * head_dim;
      accumulator =
          fmaf(scores[token - first_token], ToFloat(value[offset + dimension]), accumulator);
    }
    partials[output_offset + dimension] = accumulator;
  }
  if (threadIdx.x == 0) {
    partials[output_offset + head_dim] = maximum;
    partials[output_offset + head_dim + 1] = denominator;
  }
}

template <typename T>
__global__ void MergeDecodeAttentionImpl(const float *const partials, const int32_t heads,
                                         const int32_t partitions, const int32_t head_dim,
                                         T *const output) {
  const int32_t head = blockIdx.x;
  const int32_t dimension = threadIdx.x;
  if (dimension >= head_dim) {
    return;
  }
  float maximum = -INFINITY;
  float denominator = 0.0F;
  float accumulator = 0.0F;
  for (int32_t partition = 0; partition < partitions; ++partition) {
    const int64_t offset = (static_cast<int64_t>(partition) * heads + head) * (head_dim + 2);
    const float next_maximum = fmaxf(maximum, partials[offset + head_dim]);
    const float rescale = expf(maximum - next_maximum);
    const float probability = expf(partials[offset + head_dim] - next_maximum);
    accumulator = accumulator * rescale + probability * partials[offset + dimension];
    denominator = denominator * rescale + probability * partials[offset + head_dim + 1];
    maximum = next_maximum;
  }
  output[head * head_dim + dimension] = FromFloat<T>(accumulator / denominator);
}

} // namespace gqa_attention_kernel
} // namespace kernels
} // namespace omni_runtime
