#include <cfloat>
#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace gqa_attention_kernel {

constexpr int32_t kMaxHeadDim = 256;
constexpr int32_t kMaxValuesPerThread = 2;

template <typename T>
__device__ void
GqaAttentionImpl(const T *const __restrict__ query, const T *const __restrict__ key_cache,
                 const T *const __restrict__ value_cache, const int32_t query_token_count,
                 const int32_t query_head_count, const int32_t key_value_head_count,
                 const int32_t cached_token_count, const int32_t start_position,
                 const int32_t head_dim, const bool is_causal, T *const __restrict__ output) {
  const int32_t query_row = static_cast<int32_t>(blockIdx.x);
  if (query_row >= query_token_count * query_head_count ||
      query_head_count % key_value_head_count != 0 || head_dim > kMaxHeadDim || blockDim.x < 128) {
    return;
  }

  const int32_t query_token = query_row / query_head_count;
  const int32_t query_head = query_row - query_token * query_head_count;
  const int32_t heads_per_group = query_head_count / key_value_head_count;
  const int32_t key_value_head = query_head / heads_per_group;
  const int32_t global_position = start_position + query_token;
  const int32_t source_limit =
      is_causal ? min(cached_token_count, global_position + 1) : cached_token_count;
  if (source_limit <= 0) {
    return;
  }

  const int64_t query_offset = static_cast<int64_t>(query_row) * head_dim;
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
    const int64_t cache_offset =
        (static_cast<int64_t>(source_token) * key_value_head_count + key_value_head) * head_dim;
    float partial_score = 0.0f;
#pragma unroll
    for (int32_t item = 0; item < kMaxValuesPerThread; ++item) {
      const int32_t dimension = dimensions[item];
      if (dimension >= 0) {
        partial_score +=
            ToFloat(query[query_offset + dimension]) * ToFloat(key_cache[cache_offset + dimension]);
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

#pragma unroll
    for (int32_t item = 0; item < kMaxValuesPerThread; ++item) {
      const int32_t dimension = dimensions[item];
      if (dimension >= 0) {
        accumulator[item] = accumulator[item] * previous_scale +
                            probability_scale * ToFloat(value_cache[cache_offset + dimension]);
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

} // namespace gqa_attention_kernel
} // namespace kernels
} // namespace omni_runtime

