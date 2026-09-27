#pragma once

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime::kernels::vision_encoder_kernel {

template <typename T, int32_t kIsFp16KeyValue>
__device__ __forceinline__ void LoadVisionHalfTile(const void *const source, const int32_t tokens,
                                                   const int32_t heads, const int32_t head,
                                                   const int32_t first, __half *const tile) {
  for (int32_t index = threadIdx.x; index < 64 * 9; index += blockDim.x) {
    const int32_t row = index / 9;
    const int32_t d = index % 9 * 8;
    const int64_t offset = (static_cast<int64_t>(first + row) * heads + head) * 72 + d;
    if (first + row >= tokens) {
      reinterpret_cast<uint4 *>(tile)[index] = make_uint4(0U, 0U, 0U, 0U);
    } else if (kIsFp16KeyValue != 0 && (reinterpret_cast<uintptr_t>(source) & 15U) == 0U) {
      reinterpret_cast<uint4 *>(tile)[index] =
          *reinterpret_cast<const uint4 *>(static_cast<const __half *>(source) + offset);
    } else {
#pragma unroll
      for (int32_t item = 0; item < 8; ++item) {
        if constexpr (kIsFp16KeyValue != 0) {
          tile[index * 8 + item] = static_cast<const __half *>(source)[offset + item];
        } else {
          tile[index * 8 + item] =
              __float2half_rn(ToFloat(static_cast<const T *>(source)[offset + item]));
        }
      }
    }
  }
}

template <typename T, int32_t kIsFp16KeyValue, int32_t kQueries, int32_t kWarpQueries>
__global__ __launch_bounds__(kQueries / kWarpQueries * 32, 2) void VisionHalfAttention(
    const T *const query, const void *const key, const void *const value, const int32_t tokens,
    const int32_t heads, T *const output) {
  constexpr int32_t kKeys = 64;
  constexpr int32_t kHead = 72;
  const int32_t tiles = (tokens + kQueries - 1) / kQueries;
  const int32_t first_query = blockIdx.x % tiles * kQueries;
  const int32_t head = blockIdx.x / tiles;
  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  __shared__ __align__(16) __half queries[kQueries][kHead];
  __shared__ __align__(16) __half keys[kKeys][kHead];
  __shared__ __align__(16) __half probabilities[kQueries / kWarpQueries][kKeys][kWarpQueries];
  __half2 accumulators[2][kWarpQueries][2] = {};
  float maxima[2][kWarpQueries];
  float denominators[2][kWarpQueries] = {};
#pragma unroll
  for (int32_t part = 0; part < 2; ++part) {
#pragma unroll
    for (int32_t q = 0; q < kWarpQueries; ++q) {
      maxima[part][q] = -INFINITY;
    }
  }
  const float scale = __fdiv_rn(1.0F, __fsqrt_rn(static_cast<float>(kHead)));
  for (int32_t index = threadIdx.x; index < kQueries * kHead; index += blockDim.x) {
    const int32_t row = index / kHead;
    const int32_t d = index % kHead;
    const int64_t offset = (static_cast<int64_t>(first_query + row) * heads + head) * kHead + d;
    queries[row][d] =
        first_query + row < tokens
            ? __hmul(__float2half_rn(ToFloat(query[offset]) * scale), __float2half_rn(0.25F))
            : __float2half_rn(0.0F);
  }
  // Interleaved key streams keep the FP16 value accumulators within range.
  for (int32_t first = 0; first < tokens; first += 2 * kKeys) {
#pragma unroll
    for (int32_t part = 0; part < 2; ++part) {
      const int32_t key_first = first + part * kKeys;
      if (key_first < tokens) {
        LoadVisionHalfTile<T, kIsFp16KeyValue>(key, tokens, heads, head, key_first, &keys[0][0]);
        __syncthreads();
        float scores[kWarpQueries][2] = {};
        for (int32_t d = 0; d < kHead; d += 8) {
          const uint4 packed_keys0 = *reinterpret_cast<const uint4 *>(&keys[lane][d]);
          const uint4 packed_keys1 = *reinterpret_cast<const uint4 *>(&keys[lane + 32][d]);
#pragma unroll
          for (int32_t q = 0; q < kWarpQueries; ++q) {
            const int32_t row = warp * kWarpQueries + q;
            const uint4 packed_query = *reinterpret_cast<const uint4 *>(&queries[row][d]);
#pragma unroll
            for (int32_t pair = 0; pair < 4; ++pair) {
              const __half2 x = reinterpret_cast<const __half2 *>(&packed_query)[pair];
              const __half2 k0 = reinterpret_cast<const __half2 *>(&packed_keys0)[pair];
              const __half2 k1 = reinterpret_cast<const __half2 *>(&packed_keys1)[pair];
              const float2 first_product = __half22float2(__hmul2_rn(x, k0));
              const float2 second_product = __half22float2(__hmul2_rn(x, k1));
              scores[q][0] += first_product.x + first_product.y;
              scores[q][1] += second_product.x + second_product.y;
            }
          }
        }
        __syncthreads();
        LoadVisionHalfTile<T, kIsFp16KeyValue>(value, tokens, heads, head, key_first, &keys[0][0]);
#pragma unroll
        for (int32_t q = 0; q < kWarpQueries; ++q) {
          float maximum = maxima[part][q];
          for (int32_t item = 0; item < 2; ++item) {
            scores[q][item] =
                key_first + lane + item * 32 < tokens ? scores[q][item] * 4.0F : -INFINITY;
            maximum = fmaxf(maximum, scores[q][item] + 3.0F * 0.6931F);
          }
          for (int32_t delta = 16; delta > 0; delta /= 2) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xFFFFFFFFU, maximum, delta));
          }
          const float rescale = isfinite(maximum) ? __expf(maxima[part][q] - maximum) : 0.0F;
          const __half2 rounded_rescale = __float2half2_rn(rescale);
          float probability_sum = 0.0F;
          for (int32_t item = 0; item < 2; ++item) {
            const float probability = isfinite(maximum) ? __expf(scores[q][item] - maximum) : 0.0F;
            probability_sum += probability;
            probabilities[warp][lane + item * 32][q] = __float2half_rn(probability);
            accumulators[part][q][item] = __hmul2_rn(accumulators[part][q][item], rounded_rescale);
          }
          denominators[part][q] = fmaf(denominators[part][q], rescale, probability_sum);
          maxima[part][q] = maximum;
        }
        __syncthreads();
        for (int32_t source = 0; source < kKeys; ++source) {
          const int32_t d = lane * 4;
          if (d < kHead) {
            const uint2 packed_value = *reinterpret_cast<const uint2 *>(&keys[source][d]);
            __align__(16) uint32_t packed_probabilities[kWarpQueries / 2];
            if constexpr (kWarpQueries == 8) {
              *reinterpret_cast<uint4 *>(packed_probabilities) =
                  *reinterpret_cast<const uint4 *>(probabilities[warp][source]);
            } else if constexpr (kWarpQueries == 4) {
              *reinterpret_cast<uint2 *>(packed_probabilities) =
                  *reinterpret_cast<const uint2 *>(probabilities[warp][source]);
            } else {
              static_assert(kWarpQueries == 2);
              packed_probabilities[0] =
                  *reinterpret_cast<const uint32_t *>(probabilities[warp][source]);
            }
#pragma unroll
            for (int32_t q = 0; q < kWarpQueries; ++q) {
              const __half2 p = __half2half2(__ushort_as_half(
                  static_cast<uint16_t>(packed_probabilities[q / 2] >> ((q % 2) * 16))));
#pragma unroll
              for (int32_t item = 0; item < 2; ++item) {
                const __half2 v = reinterpret_cast<const __half2 *>(&packed_value)[item];
                accumulators[part][q][item] = __hfma2(v, p, accumulators[part][q][item]);
              }
            }
          }
        }
        __syncthreads();
      }
    }
  }
#pragma unroll
  for (int32_t q = 0; q < kWarpQueries; ++q) {
    const int32_t row = first_query + warp * kWarpQueries + q;
    float first_denominator = denominators[0][q];
    for (int32_t delta = 16; delta > 0; delta /= 2) {
      first_denominator += __shfl_xor_sync(0xFFFFFFFFU, first_denominator, delta);
    }
    const float maximum = fmaxf(maxima[0][q], maxima[1][q]);
    const float first_scale = __expf(maxima[0][q] - maximum);
    const float second_scale = __expf(maxima[1][q] - maximum);
    float second_denominator = denominators[1][q];
    for (int32_t delta = 16; delta > 0; delta /= 2) {
      second_denominator += __shfl_xor_sync(0xFFFFFFFFU, second_denominator, delta);
    }
    const float denominator =
        fmaf(second_scale, second_denominator, first_scale * first_denominator);
#pragma unroll
    for (int32_t item = 0; item < 2; ++item) {
      const int32_t d = lane * 4 + item * 2;
      if (row < tokens && d < kHead) {
        const float2 a = __half22float2(accumulators[0][q][item]);
        const float2 b = __half22float2(accumulators[1][q][item]);
        const int64_t offset = (static_cast<int64_t>(row) * heads + head) * kHead + d;
        output[offset] =
            FromFloat<T>(__fdividef(fmaf(second_scale, b.x, first_scale * a.x), denominator));
        output[offset + 1] =
            FromFloat<T>(__fdividef(fmaf(second_scale, b.y, first_scale * a.y), denominator));
      }
    }
  }
}

} // namespace omni_runtime::kernels::vision_encoder_kernel
