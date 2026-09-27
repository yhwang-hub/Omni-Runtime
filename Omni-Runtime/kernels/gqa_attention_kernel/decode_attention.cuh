#pragma once

#include <cfloat>
#include <type_traits>

#include <kernels/common/kernel_type_utils.h>
#include <kernels/matmul_kernel/tensor_core.cuh>

namespace omni_runtime {
namespace kernels {
namespace gqa_attention_kernel {

template <typename T>
__device__ __forceinline__ void LoadGgmlDecodeCache(const T *const source, const int32_t first,
                                                   const int32_t count, const int32_t kv_heads,
                                                   const int32_t head, __half *const cache) {
  for (int32_t index = threadIdx.x; index < 64 * 64; index += 128) {
    const int32_t token = first + index / 64;
    const int32_t d = index % 64 * 4;
    const int64_t offset = (static_cast<int64_t>(token) * kv_heads + head) * 256 + d;
    if constexpr (std::is_same_v<T, __half>) {
      if ((reinterpret_cast<uintptr_t>(source) & 7U) == 0U) {
        *reinterpret_cast<uint2 *>(cache + index / 64 * 264 + d) =
            token < count ? *reinterpret_cast<const uint2 *>(source + offset) : make_uint2(0U, 0U);
        continue;
      }
    }
    float4 values = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    if (token < count) {
      if (sizeof(T) == sizeof(float) && (reinterpret_cast<uintptr_t>(source) & 15U) == 0U) {
        values = *reinterpret_cast<const float4 *>(source + offset);
      } else {
        values = make_float4(ToFloat(source[offset]), ToFloat(source[offset + 1]),
                             ToFloat(source[offset + 2]), ToFloat(source[offset + 3]));
      }
    }
    __half2 *const destination =
        reinterpret_cast<__half2 *>(cache + index / 64 * 264 + d);
    destination[0] = __floats2half2_rn(values.x, values.y);
    destination[1] = __floats2half2_rn(values.z, values.w);
  }
}

template <typename T, int32_t kIsFp16KeyValue>
__global__ void GgmlDecodeAttentionImpl(const T *const query, const T *const key,
                                        const T *const value, const int32_t query_heads,
                                        const int32_t kv_heads, const int32_t cached_tokens,
                                        const int32_t start_position, const int32_t partition_cap,
                                        const int32_t *const cache_position,
                                        float *const partials) {
  constexpr int32_t kDimensions = 256;
  constexpr int32_t kStride = 264;
  using Cache = std::conditional_t<kIsFp16KeyValue != 0, __half, T>;
  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  const int32_t row = lane / 4;
  const int32_t column = (lane % 4) * 2;
  const int32_t kv_head = blockIdx.x;
  const int32_t part = blockIdx.y;
  const int32_t ratio = query_heads / kv_heads;
  const int32_t position = cache_position == nullptr ? start_position : *cache_position;
  const int32_t count = cache_position == nullptr ? cached_tokens : position + 1;
  const int32_t limit = min(count, position + 1);
  const int32_t iterations = ((count + 255) / 256) * 4;
  const int32_t partitions = min(partition_cap, iterations);
  if (part >= partitions) {
    return;
  }
  const int32_t begin = part * iterations / partitions * 64;
  const int32_t end = min((part + 1) * iterations / partitions * 64, limit);
  __shared__ __align__(16) __half cache[64 * kStride];
  __shared__ __align__(16) __half probabilities[64 * 8];
  __shared__ float metadata[4][8][2];
  __shared__ float combined[8][2];
  uint32_t queries[16][2];
  uint32_t accumulators[16][2] = {};
  float maxima[2] = {-FLT_MAX / 2.0F, -FLT_MAX / 2.0F};
  float denominators[2] = {};
#pragma unroll
  for (int32_t d = 0; d < kDimensions; d += 16) {
    const int32_t head = kv_head * ratio + row;
    const int64_t offset = static_cast<int64_t>(head) * kDimensions + d + column;
#pragma unroll
    for (int32_t item = 0; item < 2; ++item) {
      queries[d / 16][item] =
          row < ratio ? PackPair(__hmul(__float2half_rn(ToFloat(query[offset + item * 8])),
                                        __float2half_rn(0.0625F)),
                                 __hmul(__float2half_rn(ToFloat(query[offset + item * 8 + 1])),
                                        __float2half_rn(0.0625F)))
                      : 0U;
    }
  }
  for (int32_t first = begin; first < end; first += 64) {
    LoadGgmlDecodeCache(reinterpret_cast<const Cache *>(key), first, count, kv_heads, kv_head,
                        cache);
    __syncthreads();
    float scores[4] = {};
#pragma unroll
    for (int32_t d = 0; d < kDimensions; d += 16) {
      const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
          cache + (warp * 16 + lane % 16) * kStride + d + lane / 16 * 8));
      uint32_t a[4];
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                   : "r"(address));
      TensorMma<__half>(a, queries[d / 16], scores);
    }
    float rescales[2];
#pragma unroll
    for (int32_t item = 0; item < 2; ++item) {
      float maximum = maxima[item];
#pragma unroll
      for (int32_t source = 0; source < 2; ++source) {
        const int32_t token = first + warp * 16 + row + source * 8;
        scores[source * 2 + item] =
            token < limit && column + item < ratio ? scores[source * 2 + item] : -INFINITY;
        maximum = fmaxf(maximum, scores[source * 2 + item] + 3.0F * 0.6931F);
      }
#pragma unroll
      for (int32_t distance = 16; distance >= 4; distance >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, distance));
      }
      const float difference = maxima[item] - maximum;
      rescales[item] = difference >= -20.0F ? __expf(difference) : 0.0F;
      float sum = 0.0F;
#pragma unroll
      for (int32_t source = 0; source < 2; ++source) {
        const float probability = __expf(scores[source * 2 + item] - maximum);
        const int32_t token = warp * 16 + row + source * 8;
        probabilities[token * 8 + column + item] = __float2half_rn(probability);
        sum += probability;
      }
      denominators[item] = fmaf(denominators[item], rescales[item], sum);
      maxima[item] = maximum;
    }
#pragma unroll
    for (int32_t d = 0; d < 16; ++d) {
#pragma unroll
      for (int32_t item = 0; item < 2; ++item) {
        const uint32_t previous = accumulators[d][item];
        accumulators[d][item] = PackPair(
            __hmul(__ushort_as_half(static_cast<uint16_t>(previous)), __float2half_rn(rescales[0])),
            __hmul(__ushort_as_half(previous >> 16U), __float2half_rn(rescales[1])));
      }
    }
    __syncthreads();
    LoadGgmlDecodeCache(reinterpret_cast<const Cache *>(value), first, count, kv_heads, kv_head,
                        cache);
    __syncthreads();
    const uint32_t probability_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(probabilities + (warp * 16 + lane % 16) * 8));
    uint32_t b[2];
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1])
                 : "r"(probability_address));
#pragma unroll
    for (int32_t d = 0; d < kDimensions; d += 16) {
      const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
          cache + (warp * 16 + lane % 8 + lane / 16 * 8) * kStride + d +
          (lane / 8 % 2) * 8));
      uint32_t a[4];
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                   : "r"(address));
      uint32_t &c0 = accumulators[d / 16][0];
      uint32_t &c1 = accumulators[d / 16][1];
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                   "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                   : "+r"(c0), "+r"(c1)
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    __syncthreads();
  }
#pragma unroll
  for (int32_t item = 0; item < 2; ++item) {
#pragma unroll
    for (int32_t distance = 16; distance >= 4; distance >>= 1) {
      denominators[item] += __shfl_xor_sync(0xffffffffU, denominators[item], distance);
    }
    if (row == 0) {
      metadata[warp][column + item][0] = maxima[item];
      metadata[warp][column + item][1] = denominators[item];
    }
  }
#pragma unroll
  for (int32_t d = 0; d < kDimensions; d += 16) {
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t head = column + item % 2;
      const int32_t dimension = d + row + item / 2 * 8;
      const uint32_t bits = accumulators[d / 16][item / 2] >> (item % 2 * 16);
      cache[(warp * 8 + head) * kDimensions + dimension] =
          __ushort_as_half(static_cast<uint16_t>(bits));
    }
  }
  __syncthreads();
  if (threadIdx.x < 8) {
    const int32_t head = threadIdx.x;
    float maximum = metadata[0][head][0];
#pragma unroll
    for (int32_t stream = 1; stream < 4; ++stream) {
      maximum = fmaxf(maximum, metadata[stream][head][0]);
    }
    float sums[4];
#pragma unroll
    for (int32_t stream = 0; stream < 4; ++stream) {
      const float scale = __expf(metadata[stream][head][0] - maximum);
      sums[stream] = __fmul_rn(scale, metadata[stream][head][1]);
      metadata[stream][head][0] = scale;
    }
    combined[head][0] = maximum;
    combined[head][1] = __fadd_rn(__fadd_rn(sums[0], sums[2]), __fadd_rn(sums[1], sums[3]));
  }
  __syncthreads();
  for (int32_t index = threadIdx.x; index < ratio * kDimensions; index += 128) {
    const int32_t head = index / kDimensions;
    const int32_t d = index % kDimensions;
    float sum = 0.0F;
#pragma unroll
    for (int32_t stream = 0; stream < 4; ++stream) {
      sum = fmaf(__half2float(cache[(stream * 8 + head) * kDimensions + d]),
                 metadata[stream][head][0], sum);
    }
    const int64_t offset =
        (static_cast<int64_t>(part) * query_heads + kv_head * ratio + head) * (kDimensions + 2);
    partials[offset + d] = sum;
    if (d == 0) {
      partials[offset + kDimensions] = combined[head][0];
      partials[offset + kDimensions + 1] = combined[head][1];
    }
  }
}

template <typename T>
__global__ void MergeGgmlDecodeAttentionImpl(const float *const partials, const int32_t query_heads,
                                             const int32_t cached_tokens,
                                             const int32_t partition_cap,
                                             const int32_t *const cache_position, T *const output) {
  const int32_t count = cache_position == nullptr ? cached_tokens : *cache_position + 1;
  const int32_t partitions = min(partition_cap, ((count + 255) / 256) * 4);
  const int32_t head = blockIdx.x;
  const int32_t d = threadIdx.x;
  int64_t offset = (static_cast<int64_t>(partitions - 1) * query_heads + head) * 258;
  float numerator = partials[offset + d];
  float maximum = partials[offset + 256];
  float denominator = partials[offset + 257];
  for (int32_t part = partitions - 2; part >= 0; --part) {
    offset = (static_cast<int64_t>(part) * query_heads + head) * 258;
    const float next = fmaxf(maximum, partials[offset + 256]);
    const float difference = maximum - next;
    const float addition_difference = partials[offset + 256] - next;
    const float scale = difference >= -20.0F ? __expf(difference) : 0.0F;
    const float addition_scale = addition_difference >= -20.0F ? __expf(addition_difference) : 0.0F;
    numerator = fmaf(scale, numerator, addition_scale * partials[offset + d]);
    denominator = fmaf(scale, denominator, addition_scale * partials[offset + 257]);
    maximum = next;
  }
  output[static_cast<int64_t>(head) * 256 + d] = FromFloat<T>(__fdividef(numerator, denominator));
}

template <typename T>
__global__ void GroupedDecodeAttentionImpl(const T *const query, const T *const key,
                                           const T *const value, const int32_t query_heads,
                                           const int32_t kv_heads, const int32_t cached_tokens,
                                           const int32_t head_dim, float *const partials,
                                           const int32_t *const cache_position) {
  constexpr int32_t kPartition = 128;
  constexpr int32_t kStride = 136;
  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  const int32_t row = lane / 4;
  const int32_t column = (lane % 4) * 2;
  const int32_t kv_head = blockIdx.x;
  const int32_t first_token = blockIdx.y * kPartition;
  const int32_t token_count = cache_position == nullptr ? cached_tokens : *cache_position + 1;
  if (first_token >= token_count) {
    return;
  }
  const int32_t count = min(kPartition, token_count - first_token);
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
                                    float *const partials, const int32_t *const cache_position) {
  constexpr int32_t kPartition = 128;
  const int32_t head = blockIdx.x;
  const int32_t partition = blockIdx.y;
  const int32_t lane = threadIdx.x % 8;
  const int32_t kv_head = head / (query_heads / kv_heads);
  __shared__ float scores[kPartition];
  __shared__ float scratch[4];
  const int32_t first_token = partition * kPartition;
  const int32_t token_count = cache_position == nullptr ? cached_tokens : *cache_position + 1;
  if (first_token >= token_count) {
    return;
  }
  const int32_t limit = min(token_count, (partition + 1) * kPartition);
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
                                         T *const output, const int32_t *const cache_position) {
  const int32_t head = blockIdx.x;
  const int32_t dimension = threadIdx.x;
  if (dimension >= head_dim) {
    return;
  }
  float maximum = -INFINITY;
  float denominator = 0.0F;
  float accumulator = 0.0F;
  const int32_t active_partitions =
      cache_position == nullptr ? partitions : (*cache_position + 128) / 128;
  for (int32_t partition = 0; partition < active_partitions; ++partition) {
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
