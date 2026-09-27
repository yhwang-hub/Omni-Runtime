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
__device__ void
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

  for (int32_t neuron = static_cast<int32_t>(threadIdx.x); neuron < intermediate_size;
       neuron += static_cast<int32_t>(blockDim.x)) {
    float value = ToFloat(fc1_bias[neuron]);
    const int64_t weight_offset = static_cast<int64_t>(neuron) * input_size;
    for (int32_t index = 0; index < input_size; ++index) {
      value = fmaf(input_cache[index], ToFloat(fc1_weight[weight_offset + index]), value);
    }
    intermediate[neuron] = GeluTanh(value);
  }
  __syncthreads();

  const int64_t output_offset = static_cast<int64_t>(row) * output_size;
  for (int32_t neuron = static_cast<int32_t>(threadIdx.x); neuron < output_size;
       neuron += static_cast<int32_t>(blockDim.x)) {
    float value = ToFloat(fc2_bias[neuron]);
    const int64_t weight_offset = static_cast<int64_t>(neuron) * intermediate_size;
    for (int32_t index = 0; index < intermediate_size; ++index) {
      value = fmaf(intermediate[index], ToFloat(fc2_weight[weight_offset + index]), value);
    }
    output[output_offset + neuron] = FromFloat<T>(value);
  }
}

} // namespace fused_ffn_kernel
} // namespace kernels
} // namespace omni_runtime

