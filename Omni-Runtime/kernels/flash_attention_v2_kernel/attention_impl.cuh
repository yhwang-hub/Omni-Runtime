#pragma once

#include <type_traits>

#include <kernels/common/kernel_type_utils.h>
#include <kernels/matmul_kernel/tensor_core.cuh>

namespace omni_runtime::kernels::flash_attention_v2_kernel {

struct SynchronousStaging final {
  template <typename T, int32_t kDimensions, int32_t kKeys, typename S>
  static __device__ void Load(const S *const key, const S *const value, const int32_t first_key,
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
        if constexpr (std::is_same_v<T, S>) {
          keys[destination + item] = has_value ? key[offset + item] : FromFloat<T>(0.0F);
          values[destination + item] = has_value ? value[offset + item] : FromFloat<T>(0.0F);
        } else {
          keys[destination + item] =
              has_value ? FromFloat<T>(ToFloat(key[offset + item])) : FromFloat<T>(0.0F);
          values[destination + item] =
              has_value ? FromFloat<T>(ToFloat(value[offset + item])) : FromFloat<T>(0.0F);
        }
      }
    }
  }

  static __device__ void Wait() {
    __syncthreads();
  }
};

template <typename T, int32_t kDimensions, int32_t kKeys, int32_t kStages, typename Staging,
          int32_t kIsGgml, typename S, typename C>
__device__ void RunTensorAttention(const S *const query, const C *const key, const C *const value,
                                   const int32_t query_tokens, const int32_t query_heads,
                                   const int32_t kv_heads, const int32_t cached_tokens,
                                   const int32_t start_position, const int32_t head_dim,
                                   const bool is_causal, const int32_t stream_groups,
                                   S *const output) {
  constexpr int32_t kQueries = kIsGgml != 0 ? 8 : kDimensions <= 80 ? 128 : 64;
  constexpr int32_t kItems = kIsGgml != 0 ? 2 : 4;
  constexpr int32_t kRows = kIsGgml != 0 ? 1 : 2;
  constexpr int32_t kStride = kDimensions + 8;
  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  const int32_t row = lane / 4;
  const int32_t column = (lane % 4) * 2;
  const int32_t head = blockIdx.y;
  const int32_t kv_head = head / (query_heads / kv_heads);
  const int32_t first_query = blockIdx.x * kQueries + warp * 16;
  const float scale = kIsGgml != 0 ? __fdiv_rn(1.0F, __fsqrt_rn(static_cast<float>(head_dim)))
                                   : rsqrtf(static_cast<float>(head_dim));
  constexpr bool kDynamic = 2 * kStages * kKeys * kStride * sizeof(T) > 48U * 1024U;
  constexpr int32_t kStaticElements = kDynamic ? 1 : kStages * kKeys * kStride;
  __shared__ T static_keys[kStaticElements];
  __shared__ T static_values[kStaticElements];
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  T *const staged_keys = kDynamic ? reinterpret_cast<T *>(dynamic_storage) : static_keys;
  T *const staged_values = kDynamic ? staged_keys + kStages * kKeys * kStride : static_values;
  // PTX fragment layout keeps Q, scores, softmax and rescaled O in registers.
  uint32_t queries[kDimensions / 16][4];
  float accumulators[kDimensions / 8][kItems] = {};
  float merged[kDimensions / 8][kItems] = {};
  float merged_maxima[kRows];
  float merged_denominators[kRows] = {};
#pragma unroll
  for (int32_t item = 0; item < kRows; ++item) {
    merged_maxima[item] = -INFINITY;
  }
  float maxima[2] = {-INFINITY, -INFINITY};
  float denominators[2] = {};
#pragma unroll
  for (int32_t dimension = 0; dimension < kDimensions; dimension += 16) {
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t token = first_query + row + (item % 2) * 8;
      const int32_t d = dimension + column + (item / 2) * 8;
      const int64_t offset = (static_cast<int64_t>(token) * query_heads + head) * head_dim + d;
      if constexpr (kIsGgml != 0) {
        const __half rounded_scale = __float2half_rn(scale);
        queries[dimension / 16][item] =
            token < query_tokens && token < first_query + kQueries && d < head_dim
                ? PackPair(__hmul(__float2half_rn(ToFloat(query[offset])), rounded_scale),
                           d + 1 < head_dim
                               ? __hmul(__float2half_rn(ToFloat(query[offset + 1])), rounded_scale)
                               : __half{})
                : 0U;
      } else if constexpr (std::is_same_v<T, S>) {
        queries[dimension / 16][item] =
            token < query_tokens && d < head_dim
                ? PackPair(query[offset], d + 1 < head_dim ? query[offset + 1] : FromFloat<T>(0.0F))
                : 0U;
      } else {
        queries[dimension / 16][item] =
            token < query_tokens && d < head_dim
                ? PackPair(FromFloat<T>(ToFloat(query[offset])),
                           d + 1 < head_dim ? FromFloat<T>(ToFloat(query[offset + 1]))
                                            : FromFloat<T>(0.0F))
                : 0U;
      }
    }
  }
  const int32_t limit = is_causal ? min(static_cast<int64_t>(cached_tokens),
                                        static_cast<int64_t>(start_position) +
                                            static_cast<int64_t>(blockIdx.x + 1) * kQueries)
                                  : cached_tokens;
  if constexpr (kStages == 2) {
    Staging::template Load<T, kDimensions, kKeys>(key, value, 0, cached_tokens, kv_heads, kv_head,
                                                  head_dim, staged_keys, staged_values);
  }
  const int32_t parts = kIsGgml != 0 && stream_groups > 0 ? stream_groups : 1;
  int32_t first_part = 0;
  int32_t last_part = parts - 1;
  if constexpr (kIsGgml != 0) {
    if (stream_groups > 0) {
      const int64_t token_tiles = (static_cast<int64_t>(query_tokens) + 7) / 8;
      const int64_t key_tiles = ((static_cast<int64_t>(cached_tokens) + 255) / 256) * 8;
      const int64_t tile = kv_head * token_tiles + blockIdx.x;
      const int64_t total_work = token_tiles * kv_heads * key_tiles;
      first_part = static_cast<int32_t>(tile * key_tiles * stream_groups / total_work);
      last_part = static_cast<int32_t>(min(static_cast<int64_t>(parts - 1),
                                           (tile + 1) * key_tiles * stream_groups / total_work));
    }
  }
  for (int32_t part = last_part; part >= first_part; --part) {
    int32_t first_key_begin = 0;
    int32_t first_key_stop = limit;
    if constexpr (kIsGgml != 0) {
      if (stream_groups > 0) {
        const int64_t token_tiles = (static_cast<int64_t>(query_tokens) + 7) / 8;
        const int64_t key_tiles = ((static_cast<int64_t>(cached_tokens) + 255) / 256) * 8;
        const int64_t tile = kv_head * token_tiles + blockIdx.x;
        const int64_t total_work = token_tiles * kv_heads * key_tiles;
        const int64_t first = static_cast<int64_t>(part) * total_work / stream_groups;
        const int64_t stop = static_cast<int64_t>(part + 1) * total_work / stream_groups;
        const int64_t begin = max(first, tile * key_tiles);
        const int64_t end = min(stop, (tile + 1) * key_tiles);
        if (begin >= end) {
          continue;
        }
        first_key_begin = static_cast<int32_t>((begin - tile * key_tiles) * kKeys);
        first_key_stop = static_cast<int32_t>(
            min((end - tile * key_tiles) * kKeys, static_cast<int64_t>(limit)));
      }
#pragma unroll
      for (int32_t item = 0; item < kRows; ++item) {
        maxima[item] = -INFINITY;
        denominators[item] = 0.0F;
      }
#pragma unroll
      for (int32_t tile = 0; tile < kDimensions / 8; ++tile) {
#pragma unroll
        for (int32_t item = 0; item < kItems; ++item) {
          accumulators[tile][item] = 0.0F;
        }
      }
    }
    for (int32_t first_key = first_key_begin; first_key < first_key_stop; first_key += kKeys) {
      const int32_t stage = (first_key / kKeys) % kStages;
      T *const keys = staged_keys + stage * kKeys * kStride;
      T *const values = staged_values + stage * kKeys * kStride;
      if constexpr (kStages == 1) {
        Staging::template Load<T, kDimensions, kKeys>(key, value, first_key, cached_tokens,
                                                      kv_heads, kv_head, head_dim, keys, values);
      }
      Staging::Wait();
      if constexpr (kStages == 2) {
        if (first_key + kKeys < limit) {
          const int32_t next_stage = 1 - stage;
          Staging::template Load<T, kDimensions, kKeys>(
              key, value, first_key + kKeys, cached_tokens, kv_heads, kv_head, head_dim,
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
      for (int32_t item_row = 0; item_row < kRows; ++item_row) {
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
                        ? score * (kIsGgml != 0 ? 1.0F : scale)
                        : -INFINITY;
            maximum = fmaxf(maximum, score + (kIsGgml != 0 ? 3.0F * 0.6931F : 0.0F));
          }
        }
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, 1));
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, 2));
        const float rescale = !isfinite(maximum) ? 0.0F
                              : kIsGgml != 0     ? __expf(maxima[item_row] - maximum)
                                                 : expf(maxima[item_row] - maximum);
        float sum = 0.0F;
#pragma unroll
        for (int32_t tile = 0; tile < kKeys / 8; ++tile) {
#pragma unroll
          for (int32_t item = 0; item < 2; ++item) {
            const float probability = !isfinite(maximum) ? 0.0F
                                      : kIsGgml != 0
                                          ? __expf(scores[tile][item_row * 2 + item] - maximum)
                                          : expf(scores[tile][item_row * 2 + item] - maximum);
            scores[tile][item_row * 2 + item] = probability;
            sum += probability;
          }
        }
        if constexpr (kIsGgml == 0) {
          sum += __shfl_xor_sync(0xffffffffU, sum, 1);
          sum += __shfl_xor_sync(0xffffffffU, sum, 2);
        }
        denominators[item_row] = denominators[item_row] * rescale + sum;
        maxima[item_row] = maximum;
#pragma unroll
        for (int32_t tile = 0; tile < kDimensions / 8; ++tile) {
          if constexpr (kIsGgml != 0) {
            const __half rounded_rescale = __float2half_rn(rescale);
            accumulators[tile][item_row * 2] = __half2float(
                __hmul(__float2half_rn(accumulators[tile][item_row * 2]), rounded_rescale));
            accumulators[tile][item_row * 2 + 1] = __half2float(
                __hmul(__float2half_rn(accumulators[tile][item_row * 2 + 1]), rounded_rescale));
          } else {
            accumulators[tile][item_row * 2] *= rescale;
            accumulators[tile][item_row * 2 + 1] *= rescale;
          }
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
          if constexpr (kIsGgml != 0) {
            uint32_t c0 = PackPair(__float2half_rn(accumulators[tile][0]),
                                   __float2half_rn(accumulators[tile][1]));
            uint32_t c1 = 0U;
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                         "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                         : "+r"(c0), "+r"(c1)
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            const float2 first = __half22float2(__halves2half2(
                __ushort_as_half(static_cast<uint16_t>(c0)), __ushort_as_half(c0 >> 16U)));
            accumulators[tile][0] = first.x;
            accumulators[tile][1] = first.y;
          } else {
            TensorMma<T>(a, b, accumulators[tile]);
          }
        }
      }
      __syncthreads();
    }
    if constexpr (kIsGgml != 0) {
#pragma unroll
      for (int32_t item = 0; item < kRows; ++item) {
        denominators[item] += __shfl_xor_sync(0xffffffffU, denominators[item], 2);
        denominators[item] += __shfl_xor_sync(0xffffffffU, denominators[item], 1);
        const float maximum = fmaxf(merged_maxima[item], maxima[item]);
        const float old_diff = merged_maxima[item] - maximum;
        const float new_diff = maxima[item] - maximum;
        const float old_scale = old_diff >= -20.0F ? __expf(old_diff) : 0.0F;
        const float new_scale = new_diff >= -20.0F ? __expf(new_diff) : 0.0F;
        merged_denominators[item] =
            fmaf(old_scale, merged_denominators[item], new_scale * denominators[item]);
        merged_maxima[item] = maximum;
#pragma unroll
        for (int32_t tile = 0; tile < kDimensions / 8; ++tile) {
#pragma unroll
          for (int32_t column = 0; column < 2; ++column) {
            merged[tile][item * 2 + column] =
                fmaf(old_scale, merged[tile][item * 2 + column],
                     new_scale * accumulators[tile][item * 2 + column]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int32_t tile = 0; tile < kDimensions / 8; ++tile) {
#pragma unroll
    for (int32_t item = 0; item < kItems; ++item) {
      const int32_t token = first_query + row + (item / 2) * 8;
      const int32_t d = tile * 8 + column + item % 2;
      if (token < query_tokens && d < head_dim) {
        output[(static_cast<int64_t>(token) * query_heads + head) * head_dim + d] = FromFloat<S>(
            kIsGgml != 0 ? __fdividef(merged[tile][item], merged_denominators[item / 2])
                         : accumulators[tile][item] / denominators[item / 2]);
      }
    }
  }
}

} // namespace omni_runtime::kernels::flash_attention_v2_kernel
