#pragma once

#include <cfloat>
#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace attention_impl {

constexpr int32_t kMaxHeadDim = 256;
constexpr int32_t kMaxValuesPerThread = 2;

template <typename T>
__device__ void RunOnlineAttention(const T *const query, const T *const key, const T *const value,
                                   const int32_t heads, const int32_t tokens,
                                   const int32_t head_dim, const bool is_causal, T *const output) {
  const int32_t query_row = static_cast<int32_t>(blockIdx.x);
  if (query_row >= heads * tokens || head_dim > kMaxHeadDim || blockDim.x < 128) {
    return;
  }

  const int32_t head = query_row / tokens;
  const int32_t query_token = query_row - head * tokens;
  const int64_t query_offset = static_cast<int64_t>(query_row) * head_dim;
  const int32_t source_limit = is_causal ? query_token + 1 : tokens;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float accumulator[kMaxValuesPerThread] = {};
  int32_t dimensions[kMaxValuesPerThread] = {-1, -1};
#pragma unroll
  for (int32_t item = 0; item < kMaxValuesPerThread; ++item) {
    const int32_t dimension = static_cast<int32_t>(threadIdx.x) + item * blockDim.x;
    dimensions[item] = dimension < head_dim ? dimension : -1;
  }

  __shared__ float warp_values[32];
  __shared__ float previous_scale;
  __shared__ float probability_scale;
  __shared__ float denominator;
  __shared__ float maximum;
  if (threadIdx.x == 0) {
    denominator = 0.0f;
    maximum = -FLT_MAX;
  }
  __syncthreads();

  for (int32_t source_token = 0; source_token < source_limit; ++source_token) {
    const int64_t key_offset = (static_cast<int64_t>(head) * tokens + source_token) * head_dim;
    float partial_score = 0.0f;
#pragma unroll
    for (int32_t item = 0; item < kMaxValuesPerThread; ++item) {
      const int32_t dimension = dimensions[item];
      if (dimension >= 0) {
        partial_score +=
            ToFloat(query[query_offset + dimension]) * ToFloat(key[key_offset + dimension]);
      }
    }
    const float score = BlockReduceSum(partial_score, warp_values) * scale;
    if (threadIdx.x == 0) {
      const float next_maximum = fmaxf(maximum, score);
      previous_scale = expf(maximum - next_maximum);
      probability_scale = expf(score - next_maximum);
      denominator = denominator * previous_scale + probability_scale;
      maximum = next_maximum;
    }
    __syncthreads();

    const int64_t value_offset = (static_cast<int64_t>(head) * tokens + source_token) * head_dim;
#pragma unroll
    for (int32_t item = 0; item < kMaxValuesPerThread; ++item) {
      const int32_t dimension = dimensions[item];
      if (dimension >= 0) {
        accumulator[item] = accumulator[item] * previous_scale +
                            probability_scale * ToFloat(value[value_offset + dimension]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int32_t item = 0; item < kMaxValuesPerThread; ++item) {
    const int32_t dimension = dimensions[item];
    if (dimension >= 0) {
      output[query_offset + dimension] = FromFloat<T>(accumulator[item] / denominator);
    }
  }
}

} // namespace attention_impl
} // namespace kernels
} // namespace omni_runtime
