#pragma once
#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace layernorm_kernel {

template <typename T>
__device__ void RmsNormRow(const T *const input, const T *const weight, const int32_t hidden_size,
                           const float epsilon, T *const output) {
  float square_sum = 0.0F;
  for (int32_t dimension = static_cast<int32_t>(threadIdx.x); dimension < hidden_size;
       dimension += static_cast<int32_t>(blockDim.x)) {
    const float value = ToFloat(input[dimension]);
    square_sum = fmaf(value, value, square_sum);
  }
  __shared__ float warp_values[32];
  square_sum = BlockReduceSum(square_sum, warp_values);
  const float inverse_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
  for (int32_t dimension = static_cast<int32_t>(threadIdx.x); dimension < hidden_size;
       dimension += static_cast<int32_t>(blockDim.x)) {
    const float normalized = ToFloat(FromFloat<T>(ToFloat(input[dimension]) * inverse_rms));
    output[dimension] = FromFloat<T>(normalized * ToFloat(weight[dimension]));
  }
}

template <typename T>
__global__ void RmsNormImpl(const T *const __restrict__ input, const T *const __restrict__ weight,
                            const int32_t rows, const int32_t hidden_size, const float epsilon,
                            T *const __restrict__ output) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  if (row < rows) {
    const int64_t offset = static_cast<int64_t>(row) * hidden_size;
    RmsNormRow(input + offset, weight, hidden_size, epsilon, output + offset);
  }
}

template <typename T>
__global__ void ResidualRmsNormImpl(const T *const input, const T *const residual,
                                    const T *const weight, const int32_t rows,
                                    const int32_t hidden_size, const float epsilon,
                                    T *const residual_output, T *const output) {
  const int32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  float square_sum = 0.0F;
  for (int32_t d = threadIdx.x; d < hidden_size; d += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * hidden_size + d;
    const T value = FromFloat<T>(ToFloat(input[index]) + ToFloat(residual[index]));
    square_sum = fmaf(ToFloat(value), ToFloat(value), square_sum);
  }
  __shared__ float scratch[32];
  square_sum = BlockReduceSum(square_sum, scratch);
  const float inverse_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
  for (int32_t d = threadIdx.x; d < hidden_size; d += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * hidden_size + d;
    const T value = FromFloat<T>(ToFloat(input[index]) + ToFloat(residual[index]));
    residual_output[index] = value;
    const float normalized = ToFloat(FromFloat<T>(ToFloat(value) * inverse_rms));
    output[index] = FromFloat<T>(normalized * ToFloat(weight[d]));
  }
}

template <typename T>
__global__ void LayernormImpl(const T *const __restrict__ input, const T *const __restrict__ weight,
                              const T *const __restrict__ bias, const int32_t rows,
                              const int32_t hidden_size, const float epsilon,
                              T *const __restrict__ output) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
  float sum = 0.0F;
  float square_sum = 0.0F;
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
  const float variance = fmaxf(square_sum / static_cast<float>(hidden_size) - mean * mean, 0.0F);
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
