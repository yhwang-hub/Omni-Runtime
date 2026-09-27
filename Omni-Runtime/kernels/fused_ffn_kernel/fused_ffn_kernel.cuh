#pragma once
#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace fused_ffn_kernel {

__device__ inline float GeluTanh(const float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  constexpr float kBeta = 0.044715f;
  return 0.5f * value * (1.0f + tanhf(kAlpha * (value + kBeta * value * value * value)));
}

template <typename T>
__device__ float ProjectNeuron(const float *const input, const T *const weight, const int32_t width,
                               const int32_t neuron) {
  float value = 0.0F;
  const int64_t offset = static_cast<int64_t>(neuron) * width;
  for (int32_t index = threadIdx.x % 32; index < width; index += 32) {
    value = fmaf(input[index], ToFloat(weight[offset + index]), value);
  }
  return WarpReduceSum(value);
}

template <typename T>
__global__ void
FusedFfnImpl(const T *const __restrict__ input, const T *const __restrict__ fc1_weight,
             const T *const __restrict__ fc1_bias, const T *const __restrict__ fc2_weight,
             const T *const __restrict__ fc2_bias, const int32_t rows, const int32_t input_size,
             const int32_t intermediate_size, const int32_t output_size,
             T *const __restrict__ output) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  extern __shared__ float shared_values[];
  float *const input_cache = shared_values;
  float *const intermediate = shared_values + input_size;
  const int64_t input_offset = static_cast<int64_t>(row) * input_size;
  for (int32_t index = static_cast<int32_t>(threadIdx.x); index < input_size;
       index += static_cast<int32_t>(blockDim.x)) {
    input_cache[index] = ToFloat(input[input_offset + index]);
  }
  __syncthreads();

  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  const int32_t warps = blockDim.x / 32;
  for (int32_t neuron = warp; neuron < intermediate_size; neuron += warps) {
    const float value = ProjectNeuron(input_cache, fc1_weight, input_size, neuron);
    if (lane == 0) {
      intermediate[neuron] = GeluTanh(value + ToFloat(fc1_bias[neuron]));
    }
  }
  __syncthreads();

  const int64_t output_offset = static_cast<int64_t>(row) * output_size;
  for (int32_t neuron = warp; neuron < output_size; neuron += warps) {
    const float value = ProjectNeuron(intermediate, fc2_weight, intermediate_size, neuron);
    if (lane == 0) {
      output[output_offset + neuron] = FromFloat<T>(value + ToFloat(fc2_bias[neuron]));
    }
  }
}

} // namespace fused_ffn_kernel
} // namespace kernels
} // namespace omni_runtime
