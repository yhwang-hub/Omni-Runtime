#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace layernorm_kernel {

template <typename T>
__device__ void LayernormImpl(const T *const __restrict__ input, const T *const __restrict__ weight,
                              const T *const __restrict__ bias, const int32_t rows,
                              const int32_t hidden_size, const float epsilon,
                              T *const __restrict__ output) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
  float sum = 0.0f;
  float square_sum = 0.0f;
  for (int32_t dimension = static_cast<int32_t>(threadIdx.x); dimension < hidden_size;
       dimension += static_cast<int32_t>(blockDim.x)) {
    const float value = ToFloat(input[row_offset + dimension]);
    sum += value;
    square_sum = fmaf(value, value, square_sum);
  }
  __shared__ float warp_values[32];
  sum = BlockReduceSum(sum, warp_values);
  square_sum = BlockReduceSum(square_sum, warp_values);
  const float mean = sum / static_cast<float>(hidden_size);
  const float variance = fmaxf(square_sum / static_cast<float>(hidden_size) - mean * mean, 0.0f);
  const float inverse_standard_deviation = rsqrtf(variance + epsilon);
  for (int32_t dimension = static_cast<int32_t>(threadIdx.x); dimension < hidden_size;
       dimension += static_cast<int32_t>(blockDim.x)) {
    const float normalized =
        (ToFloat(input[row_offset + dimension]) - mean) * inverse_standard_deviation;
    output[row_offset + dimension] =
        FromFloat<T>(normalized * ToFloat(weight[dimension]) + ToFloat(bias[dimension]));
  }
}

} // namespace layernorm_kernel
} // namespace kernels
} // namespace omni_runtime

