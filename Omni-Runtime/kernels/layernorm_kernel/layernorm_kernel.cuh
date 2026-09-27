#pragma once
#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace layernorm_kernel {

__device__ inline float3 MergeLayerMoments(const float3 first, const float3 second) {
  const float count = first.z + second.z;
  if (count == 0.0F) {
    return make_float3(0.0F, 0.0F, 0.0F);
  }
  const float inverse = 1.0F / count;
  const float first_fraction = first.z * inverse;
  const float second_fraction = second.z * inverse;
  const float delta = first.x - second.x;
  const float change = __fmul_rn(__fmul_rn(delta, delta), second.z);
  return make_float3(fmaf(second_fraction, second.x, __fmul_rn(first_fraction, first.x)),
                      fmaf(change, first_fraction, second.y + first.y), count);
}

template <typename T>
__device__ float2 VectorLayerMoments(const T *const input, const int32_t hidden_size) {
  float3 moments = make_float3(0.0F, 0.0F, 0.0F);
  for (int32_t vector = threadIdx.x; vector < hidden_size / 4; vector += blockDim.x) {
    for (int32_t part = 0; part < 4; ++part) {
      const float value = ToFloat(input[vector * 4 + part]);
      const float delta = value - moments.x;
      const float count = moments.z + 1.0F;
      const float mean = moments.x + delta * (1.0F / count);
      moments = make_float3(mean, moments.y + delta * (value - mean), count);
    }
  }
  for (int32_t distance = 16; distance > 0; distance >>= 1) {
    const float3 other = make_float3(__shfl_down_sync(0xFFFFFFFFU, moments.x, distance),
                                     __shfl_down_sync(0xFFFFFFFFU, moments.y, distance),
                                     __shfl_down_sync(0xFFFFFFFFU, moments.z, distance));
    moments = MergeLayerMoments(moments, other);
  }
  __shared__ float3 warp_moments[4];
  const int32_t warp = threadIdx.x / 32;
  const int32_t lane = threadIdx.x % 32;
  if (lane == 0) {
    warp_moments[warp] = moments;
  }
  __syncthreads();
  for (int32_t distance = blockDim.x / 64; distance > 0; distance >>= 1) {
    if (lane == 0 && warp < distance) {
      moments = MergeLayerMoments(moments, warp_moments[warp + distance]);
      warp_moments[warp] = moments;
    }
    __syncthreads();
  }
  return make_float2(warp_moments[0].x,
                      warp_moments[0].y / static_cast<float>(hidden_size));
}

template <typename T>
__global__ void LayernormVectorImpl(const T *const input, const T *const weight,
                                    const T *const bias, const int32_t rows,
                                    const int32_t hidden_size, const float epsilon,
                                    T *const output) {
  const int32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const uint64_t base = static_cast<uint64_t>(row) * hidden_size;
  const float2 moments = VectorLayerMoments(input + base, hidden_size);
  const float inverse = rsqrtf(moments.y + epsilon);
  for (int32_t dimension = threadIdx.x; dimension < hidden_size; dimension += blockDim.x) {
    const float normalized = inverse * (ToFloat(input[base + dimension]) - moments.x);
    output[base + dimension] =
        FromFloat<T>(ToFloat(weight[dimension]) * normalized + ToFloat(bias[dimension]));
  }
}

template <int32_t kIsGgmlNormalization>
__device__ inline float RmsInverse(const float square_sum, const int32_t hidden_size,
                                   const float epsilon) {
  if constexpr (kIsGgmlNormalization != 0) {
    float inverse_size;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse_size) : "f"(static_cast<float>(hidden_size)));
    return rsqrtf(fmaf(square_sum, inverse_size, epsilon));
  }
  return rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
}

template <typename T>
__global__ void
BfloatRmsNormImpl(const T *const input, const T *const residual, const void *const weight,
                  const int32_t rows, const int32_t hidden_size, const float epsilon,
                  const bool is_float_weight, T *const residual_output, T *const output) {
  const int32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const uint64_t base = static_cast<uint64_t>(row) * hidden_size;
  const T *values = input + base;
  if (residual != nullptr) {
    for (int32_t d = threadIdx.x; d < hidden_size; d += blockDim.x) {
      residual_output[base + d] =
          FromFloat<T>(ToFloat(input[base + d]) + ToFloat(residual[base + d]));
    }
    __syncthreads();
    values = residual_output + base;
  }
  float partial[4] = {};
  for (int32_t vector = threadIdx.x; vector < hidden_size / 4; vector += blockDim.x) {
    for (int32_t part = 0; part < 4; ++part) {
      const float value = ToFloat(values[vector * 4 + part]);
      partial[part] += __fmul_rn(value, value);
    }
  }
  float square_sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
  __shared__ float scratch[512];
  scratch[threadIdx.x] = square_sum;
  __syncthreads();
  for (int32_t distance = blockDim.x / 2; distance >= 32; distance /= 2) {
    if (threadIdx.x < distance) {
      square_sum += scratch[threadIdx.x + distance];
      scratch[threadIdx.x] = square_sum;
    }
    __syncthreads();
  }
  for (int32_t distance = 16; distance > 0; distance /= 2) {
    square_sum += __shfl_down_sync(0xffffffffU, square_sum, distance);
  }
  if (threadIdx.x == 0) {
    scratch[0] = square_sum;
  }
  __syncthreads();
  // The mean and epsilon addition have separate FP32 rounding boundaries.
  const float mean = __fmul_rn(scratch[0], 1.0F / static_cast<float>(hidden_size));
  const float inverse = rsqrtf(mean + epsilon);
  for (int32_t d = threadIdx.x; d < hidden_size; d += blockDim.x) {
    const float scaled = ToFloat(values[d]) * inverse;
    const float normalized = is_float_weight ? scaled : ToFloat(FromFloat<T>(scaled));
    const float scale = is_float_weight ? static_cast<const float *>(weight)[d]
                                        : ToFloat(static_cast<const T *>(weight)[d]);
    output[base + d] = FromFloat<T>(normalized * scale);
  }
}

template <typename T, int32_t kIsGgmlNormalization>
__device__ void RmsNormRow(const T *const input, const void *const weight,
                           const int32_t hidden_size, const float epsilon, T *const output,
                           const bool is_float_weight) {
  float square_sum = 0.0F;
  for (int32_t dimension = static_cast<int32_t>(threadIdx.x); dimension < hidden_size;
       dimension += static_cast<int32_t>(blockDim.x)) {
    const float value = ToFloat(input[dimension]);
    square_sum = fmaf(value, value, square_sum);
  }
  __shared__ float warp_values[32];
  square_sum = BlockReduceSum(square_sum, warp_values);
  const float inverse_rms = RmsInverse<kIsGgmlNormalization>(square_sum, hidden_size, epsilon);
  for (int32_t dimension = static_cast<int32_t>(threadIdx.x); dimension < hidden_size;
       dimension += static_cast<int32_t>(blockDim.x)) {
    const float value = ToFloat(input[dimension]) * inverse_rms;
    const float normalized = is_float_weight ? value : ToFloat(FromFloat<T>(value));
    const float scale = is_float_weight ? static_cast<const float *>(weight)[dimension]
                                        : ToFloat(static_cast<const T *>(weight)[dimension]);
    output[dimension] = FromFloat<T>(normalized * scale);
  }
}

template <typename T, int32_t kIsGgmlNormalization>
__global__ void RmsNormImpl(const T *const __restrict__ input,
                            const void *const __restrict__ weight, const int32_t rows,
                            const int32_t hidden_size, const float epsilon,
                            T *const __restrict__ output, const bool is_float_weight) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  if (row < rows) {
    const int64_t offset = static_cast<int64_t>(row) * hidden_size;
    RmsNormRow<T, kIsGgmlNormalization>(input + offset, weight, hidden_size, epsilon,
                                        output + offset, is_float_weight);
  }
}

template <typename T, int32_t kIsGgmlNormalization>
__global__ void
ResidualRmsNormImpl(const T *const input, const T *const residual, const void *const weight,
                    const int32_t rows, const int32_t hidden_size, const float epsilon,
                    T *const residual_output, T *const output, const bool is_float_weight) {
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
  const float inverse_rms = RmsInverse<kIsGgmlNormalization>(square_sum, hidden_size, epsilon);
  for (int32_t d = threadIdx.x; d < hidden_size; d += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * hidden_size + d;
    const T value = FromFloat<T>(ToFloat(input[index]) + ToFloat(residual[index]));
    residual_output[index] = value;
    const float scaled = ToFloat(value) * inverse_rms;
    const float normalized = is_float_weight ? scaled : ToFloat(FromFloat<T>(scaled));
    const float scale = is_float_weight ? static_cast<const float *>(weight)[d]
                                        : ToFloat(static_cast<const T *>(weight)[d]);
    output[index] = FromFloat<T>(normalized * scale);
  }
}

template <typename T, int32_t kIsGgmlNormalization>
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
  float mean = sum / static_cast<float>(hidden_size);
  float variance = fmaxf(square_sum / static_cast<float>(hidden_size) - mean * mean, 0.0F);
  if constexpr (kIsGgmlNormalization != 0) {
    // GGML shares an approximate reciprocal between the mean and variance FMA.
    float inverse_size = 0.0F;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse_size) : "f"(static_cast<float>(hidden_size)));
    mean = sum * inverse_size;
    variance = fmaf(square_sum, inverse_size, -__fmul_rn(mean, mean));
  }
  const float inverse_standard_deviation = rsqrtf(variance + epsilon);
  for (int32_t dimension = static_cast<int32_t>(threadIdx.x); dimension < hidden_size;
       dimension += static_cast<int32_t>(blockDim.x)) {
    const float normalized =
        (ToFloat(input[row_offset + dimension]) - mean) * inverse_standard_deviation;
    if constexpr (kIsGgmlNormalization != 0) {
      const float weighted = __fmul_rn(normalized, ToFloat(weight[dimension]));
      output[row_offset + dimension] = FromFloat<T>(weighted + ToFloat(bias[dimension]));
    } else {
      output[row_offset + dimension] =
          FromFloat<T>(normalized * ToFloat(weight[dimension]) + ToFloat(bias[dimension]));
    }
  }
}

} // namespace layernorm_kernel
} // namespace kernels
} // namespace omni_runtime
