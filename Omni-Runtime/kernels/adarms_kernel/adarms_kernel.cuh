#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace adarms_kernel {

// Adaptive RMSNorm used by the pi0.5 action expert (adaRMS). Each row is
// RMS-normalized (variance over the hidden dimension, computed in float32) and
// then modulated by a per-hidden scale and shift derived from the timestep
// condition: output = normed * (1 + scale) + shift. The scale and shift vectors
// are broadcast across all rows (batch size one during inference).
template <typename T>
__device__ void AdaRmsNormImpl(const T *const __restrict__ input, const T *const __restrict__ scale,
                               const T *const __restrict__ shift, const int32_t rows,
                               const int32_t hidden_size, const float epsilon,
                               T *const __restrict__ output) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
  float square_sum = 0.0f;
  for (int32_t dimension = static_cast<int32_t>(threadIdx.x); dimension < hidden_size;
       dimension += static_cast<int32_t>(blockDim.x)) {
    const float value = ToFloat(input[row_offset + dimension]);
    square_sum = fmaf(value, value, square_sum);
  }
  __shared__ float warp_values[32];
  square_sum = BlockReduceSum(square_sum, warp_values);
  const float inverse_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
  for (int32_t dimension = static_cast<int32_t>(threadIdx.x); dimension < hidden_size;
       dimension += static_cast<int32_t>(blockDim.x)) {
    const float normalized = ToFloat(input[row_offset + dimension]) * inverse_rms;
    const float modulated =
        normalized * (1.0f + ToFloat(scale[dimension])) + ToFloat(shift[dimension]);
    output[row_offset + dimension] = FromFloat<T>(modulated);
  }
}

} // namespace adarms_kernel
} // namespace kernels
} // namespace omni_runtime

