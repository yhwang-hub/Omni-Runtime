#include <cfloat>
#include <cstdint>

#include <kernels/common/attention_kernel_impl.h>

namespace omni_runtime {
namespace kernels {
namespace flash_attention_kernel {

template <typename T>
__device__ void AttentionSoftmaxImpl(T *const __restrict__ scores, const int32_t rows,
                                     const int32_t columns, const float scale,
                                     const bool is_causal) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  if (row >= rows || columns <= 0) {
    return;
  }
  const int32_t query = row % columns;
  const int32_t source_limit = is_causal ? query + 1 : columns;
  const int64_t row_offset = static_cast<int64_t>(row) * columns;
  float local_maximum = -FLT_MAX;
  for (int32_t column = static_cast<int32_t>(threadIdx.x); column < source_limit;
       column += static_cast<int32_t>(blockDim.x)) {
    local_maximum = fmaxf(local_maximum, ToFloat(scores[row_offset + column]) * scale);
  }
  __shared__ float warp_values[32];
  const float maximum = BlockReduceMax(local_maximum, warp_values);
  float local_sum = 0.0f;
  for (int32_t column = static_cast<int32_t>(threadIdx.x); column < columns;
       column += static_cast<int32_t>(blockDim.x)) {
    const float probability =
        column < source_limit ? expf(ToFloat(scores[row_offset + column]) * scale - maximum) : 0.0f;
    scores[row_offset + column] = FromFloat<T>(probability);
    local_sum += probability;
  }
  const float inverse_sum = 1.0f / BlockReduceSum(local_sum, warp_values);
  for (int32_t column = static_cast<int32_t>(threadIdx.x); column < columns;
       column += static_cast<int32_t>(blockDim.x)) {
    scores[row_offset + column] = FromFloat<T>(ToFloat(scores[row_offset + column]) * inverse_sum);
  }
}

} // namespace flash_attention_kernel
} // namespace kernels
} // namespace omni_runtime

