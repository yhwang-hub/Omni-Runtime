#pragma once

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime::kernels::gqa_attention_kernel {

__device__ inline void FloatMma(const float (&a)[4], const float (&b)[2],
                                float *const accumulator) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
      : "r"(__float_as_uint(a[0])), "r"(__float_as_uint(a[1])), "r"(__float_as_uint(a[2])),
        "r"(__float_as_uint(a[3])), "r"(__float_as_uint(b[0])), "r"(__float_as_uint(b[1])));
}

__device__ inline float FloatResidual(const float value, const float high) {
  uint32_t bits = 0U;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(value - high));
  return __uint_as_float(bits);
}

// Score stages: 0 fuses QK/PV, 1 writes independent QK tiles, 2 consumes them in key order.
template <typename T, int32_t kHeadSize, int32_t kQueries, int32_t kIsCausal,
          int32_t kOutputPartitions, int32_t kIsGrouped, int32_t kScoreStage>
__global__ void
BfloatAttentionImpl(const T *const __restrict__ query, const T *const __restrict__ key,
                    const T *const __restrict__ value, const int32_t query_tokens,
                    const int32_t query_heads, const int32_t kv_heads, const int32_t cached_tokens,
                    const int32_t start_position, const int32_t *const cache_position,
                    float *const __restrict__ score_workspace, T *const __restrict__ output) {
  const int32_t thread = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = thread / 32;
  const int32_t lane = thread % 32;
  const int32_t head = static_cast<int32_t>(blockIdx.y);
  const int32_t first_query = kScoreStage == 1 ? 0 : static_cast<int32_t>(blockIdx.x) * kQueries;
  const int32_t groups = query_heads / kv_heads;
  const int32_t query_rows = kIsGrouped != 0 ? groups : query_tokens;
  const int32_t query_stride = kIsGrouped != 0 ? kHeadSize : query_heads * kHeadSize;
  const int32_t query_offset = head * (kIsGrouped != 0 ? groups : 1) * kHeadSize;
  const int32_t cache_stride = kv_heads * kHeadSize;
  const int32_t kv_head = kIsGrouped != 0 ? head : head / groups;
  const int32_t position = cache_position == nullptr ? start_position : *cache_position;
  const int32_t keys = cache_position == nullptr ? cached_tokens : position + 1;
  // Cached prefill uses an additive mask, so every cached key block participates.
  const bool has_offset_mask = position > 0 && query_tokens > 1;
  const int32_t first_key = kScoreStage == 1 ? static_cast<int32_t>(blockIdx.x) * 128 : 0;
  const int32_t limit = kScoreStage == 1 ? min(keys, first_key + 128)
                        : kIsCausal != 0 && !has_offset_mask
                            ? min(keys, position + first_query + kQueries)
                            : keys;
  if (first_key >= limit) {
    return;
  }
  const float scale = static_cast<float>(1.0 / sqrt(static_cast<double>(kHeadSize)));
  // Two warps cover up to eight grouped heads, exposing twice as many output tiles.
  constexpr int32_t kOutputWidth = kScoreStage == 2 && kOutputPartitions == 16 ? 16 : 32;
  constexpr int32_t kTiles = (kHeadSize + kOutputWidth - 1) / kOutputWidth;
  constexpr int32_t kOutputTiles = kTiles / kOutputPartitions;
  const int32_t first_output_tile = blockIdx.z * kOutputTiles;
  constexpr int32_t kResidue = kHeadSize % 32 == 0 ? 32 : kHeadSize % 32;
  // Give adjacent eight-lane row groups different shared-memory banks.
  constexpr int32_t kProbabilityStride = 136;
  __shared__ float probabilities[kQueries * kProbabilityStride];
  __shared__ float maximum[kQueries];
  __shared__ float denominator[kQueries];
  __shared__ float rescale[kQueries];
  __shared__ float query_tile[kIsCausal == 0 ? kQueries : 1][kIsCausal == 0 ? kHeadSize + 4 : 1];
  if constexpr (kIsCausal == 0) {
    for (int32_t index = thread; index < kQueries * kHeadSize; index += kQueries * 8) {
      const int32_t row = index / kHeadSize;
      const int32_t dimension = index % kHeadSize;
      query_tile[row][dimension] =
          first_query + row < query_rows
              ? ToFloat(query[(first_query + row) * query_stride + query_offset + dimension])
              : 0.0F;
    }
  }
  if (thread < kQueries) {
    maximum[thread] = -INFINITY;
    denominator[thread] = first_query + thread < query_rows ? 0.0F : 1.0F;
    rescale[thread] = 0.0F;
  }
  __syncthreads();
  float accumulator[kOutputTiles][4] = {};
  for (int32_t first = first_key; first < limit; first += 128) {
    if constexpr (kScoreStage == 2) {
      for (int32_t index = thread; index < kQueries * 128; index += blockDim.x) {
        const int32_t row = index / 128;
        const int32_t column = index % 128;
        probabilities[row * kProbabilityStride + column] =
            row >= groups ? 0.0F
            : first + column < keys
                ? score_workspace[static_cast<uint64_t>(head * groups + row) * cached_tokens +
                                  first + column]
                : -INFINITY;
      }
    } else {
      float scores[4][4] = {};
      float score_partials[4][4] = {};
      for (int32_t depth = 0; depth < kHeadSize; depth += 8) {
        float a[4];
        // Preserve the reference residue/full-tile accumulation boundaries.
        for (int32_t item = 0; item < 4; ++item) {
          const int32_t row = first_query + warp / 4 * 16 + lane / 4 + item % 2 * 8;
          const int32_t dimension = depth + lane % 4 + item / 2 * 4;
          if constexpr (kIsCausal == 0) {
            a[item] = dimension < kHeadSize ? query_tile[row - first_query][dimension] : 0.0F;
          } else {
            a[item] = row < query_rows && dimension < kHeadSize
                          ? ToFloat(query[row * query_stride + query_offset + dimension])
                          : 0.0F;
          }
        }
        for (int32_t tile = 0; tile < 4; ++tile) {
          float b[2];
          for (int32_t item = 0; item < 2; ++item) {
            const int32_t row = first + warp % 4 * 32 + tile * 8 + lane / 4;
            const int32_t dimension = depth + lane % 4 + item * 4;
            b[item] = row < keys && dimension < kHeadSize
                          ? ToFloat(key[row * cache_stride + kv_head * kHeadSize + dimension])
                          : 0.0F;
          }
          FloatMma(a, b, score_partials[tile]);
          if (depth == 0 || (depth >= kResidue && (depth - kResidue) % 32 == 0)) {
            for (int32_t item = 0; item < 4; ++item) {
              scores[tile][item] += score_partials[tile][item];
              score_partials[tile][item] = 0.0F;
            }
          }
        }
      }
      for (int32_t tile = 0; tile < 4; ++tile) {
        for (int32_t item = 0; item < 4; ++item) {
          const int32_t row = warp / 4 * 16 + lane / 4 + item / 2 * 8;
          const int32_t column = warp % 4 * 32 + tile * 8 + lane % 4 * 2 + item % 2;
          const float score =
              first_query + row >= query_rows ? 0.0F
              : first + column < keys &&
                      (kIsCausal == 0 ||
                       first + column <= position + (kIsGrouped != 0 ? 0 : first_query + row))
                  ? __fmul_rn(scores[tile][item] + score_partials[tile][item], scale) *
                        1.4426950408889634F
                  : -INFINITY;
          if constexpr (kScoreStage == 1) {
            if (row < groups && first + column < keys) {
              score_workspace[static_cast<uint64_t>(head * groups + row) * cached_tokens + first +
                              column] = score;
            }
          } else {
            probabilities[row * kProbabilityStride + column] = score;
          }
        }
      }
    }
    if constexpr (kScoreStage == 1) {
      return;
    }
    __syncthreads();
    {
      const int32_t row = warp * 4 + lane / 8;
      const int32_t column_lane = lane % 8;
      const uint32_t row_mask = 0xffU << (lane / 8 * 8);
      if (first_query + row < query_rows) {
        float next_maximum = maximum[row];
        for (int32_t column = column_lane; column < 128; column += 8) {
          next_maximum = fmaxf(next_maximum, probabilities[row * kProbabilityStride + column]);
        }
        for (int32_t distance = 4; distance > 0; distance /= 2) {
          next_maximum = fmaxf(next_maximum, __shfl_xor_sync(row_mask, next_maximum, distance, 8));
        }
        const float factor = exp2f(maximum[row] - next_maximum);
        for (int32_t column = column_lane; column < 128; column += 8) {
          probabilities[row * kProbabilityStride + column] =
              exp2f(probabilities[row * kProbabilityStride + column] - next_maximum);
        }
        __syncwarp(row_mask);
        float sums[2] = {};
        for (int32_t half = 0; half < 2; ++half) {
          const int32_t original_lane = column_lane + half * 8;
          for (int32_t tile = 0; tile < 4; ++tile) {
            for (int32_t item = 0; item < 2; ++item) {
              sums[half] += probabilities[row * kProbabilityStride + original_lane / 4 * 32 +
                                          tile * 8 + original_lane % 4 * 2 + item];
            }
          }
          sums[half] += __shfl_xor_sync(row_mask, sums[half], 1, 4);
          sums[half] += __shfl_xor_sync(row_mask, sums[half], 2, 4);
        }
        float total = __fmul_rn(denominator[row], factor);
        for (int32_t group = 0; group < 4; ++group) {
          total += __shfl_sync(row_mask, sums[group / 2], group % 2 * 4, 8);
        }
        __syncwarp(row_mask);
        if (column_lane == 0) {
          maximum[row] = next_maximum;
          denominator[row] = total;
          rescale[row] = factor;
        }
      }
    }
    __syncthreads();
    float block_sum[kOutputTiles][4] = {};
    if constexpr (kIsCausal == 0) {
      for (int32_t tile = 0; tile < kOutputTiles; ++tile) {
        for (int32_t item = 0; item < 4; ++item) {
          block_sum[tile][item] =
              accumulator[tile][item] * rescale[warp / 4 * 16 + lane / 4 + item / 2 * 8];
        }
      }
    }
    float partial[kOutputTiles][4] = {};
    for (int32_t depth = 0; depth < 128; depth += 8) {
      float high[4];
      float low[4];
      for (int32_t item = 0; item < 4; ++item) {
        const float probability =
            probabilities[(warp / 4 * 16 + lane / 4 + item % 2 * 8) * kProbabilityStride + depth +
                          lane % 4 + item / 2 * 4];
        high[item] = __uint_as_float(__float_as_uint(probability) & 0xffffe000U);
        low[item] = FloatResidual(probability, high[item]);
      }
      for (int32_t tile = 0; tile < kOutputTiles; ++tile) {
        const int32_t column =
            warp % 4 * 8 + (first_output_tile + tile) * kOutputWidth + lane / 4;
        float b[2];
        for (int32_t item = 0; item < 2; ++item) {
          const int32_t row = first + depth + lane % 4 + item * 4;
          b[item] = row < keys && column < kHeadSize
                        ? ToFloat(value[row * cache_stride + kv_head * kHeadSize + column])
                        : 0.0F;
        }
        FloatMma(low, b, partial[tile]);
        FloatMma(high, b, partial[tile]);
        if (depth % 32 == 0) {
          for (int32_t item = 0; item < 4; ++item) {
            block_sum[tile][item] += partial[tile][item];
            partial[tile][item] = 0.0F;
          }
        }
      }
    }
    for (int32_t tile = 0; tile < kOutputTiles; ++tile) {
      for (int32_t item = 0; item < 4; ++item) {
        block_sum[tile][item] += partial[tile][item];
        if constexpr (kIsCausal != 0) {
          const int32_t row = warp / 4 * 16 + lane / 4 + item / 2 * 8;
          const float inverse = first + 128 >= limit ? 1.0F / denominator[row] : 1.0F;
          // The global-memory epilogue normalizes the last block before merging it.
          accumulator[tile][item] =
              first == 0 ? inverse * block_sum[tile][item]
              : first + 128 >= limit
                  ? fmaf(inverse, block_sum[tile][item],
                         __fmul_rn(__fmul_rn(inverse, rescale[row]), accumulator[tile][item]))
                  : fmaf(rescale[row], accumulator[tile][item], block_sum[tile][item]);
        } else {
          accumulator[tile][item] = block_sum[tile][item];
        }
      }
    }
    __syncthreads();
  }
  for (int32_t tile = 0; tile < kOutputTiles; ++tile) {
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t row = warp / 4 * 16 + lane / 4 + item / 2 * 8;
      const int32_t column =
          warp % 4 * 8 + (first_output_tile + tile) * kOutputWidth + lane % 4 * 2 + item % 2;
      if (first_query + row < query_rows && column < kHeadSize) {
        output[(first_query + row) * query_stride + query_offset + column] =
            FromFloat<T>(kIsCausal != 0 ? accumulator[tile][item]
                                        : accumulator[tile][item] * (1.0F / denominator[row]));
      }
    }
  }
}

} // namespace omni_runtime::kernels::gqa_attention_kernel
