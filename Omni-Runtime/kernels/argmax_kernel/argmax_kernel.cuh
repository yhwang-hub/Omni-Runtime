#include <cfloat>
#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace argmax_kernel {

template <typename T>
__device__ void ArgmaxImpl(const T *const __restrict__ logits, const int32_t rows,
                           const int32_t columns, int32_t *const __restrict__ output) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  float best_value = -FLT_MAX;
  int32_t best_index = 0;
  const int64_t row_offset = static_cast<int64_t>(row) * columns;
  for (int32_t column = static_cast<int32_t>(threadIdx.x); column < columns;
       column += static_cast<int32_t>(blockDim.x)) {
    const float value = ToFloat(logits[row_offset + column]);
    if (value > best_value || (value == best_value && column < best_index)) {
      best_value = value;
      best_index = column;
    }
  }
  for (int32_t offset = 16; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(0xffffffffU, best_value, offset);
    const int32_t other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
    if (other_value > best_value || (other_value == best_value && other_index < best_index)) {
      best_value = other_value;
      best_index = other_index;
    }
  }
  __shared__ float warp_values[32];
  __shared__ int32_t warp_indices[32];
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
  const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
  if (lane == 0) {
    warp_values[warp] = best_value;
    warp_indices[warp] = best_index;
  }
  __syncthreads();
  if (warp == 0) {
    const int32_t warp_count = (static_cast<int32_t>(blockDim.x) + 31) / 32;
    best_value = lane < warp_count ? warp_values[lane] : -FLT_MAX;
    best_index = lane < warp_count ? warp_indices[lane] : 0;
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
      const float other_value = __shfl_down_sync(0xffffffffU, best_value, offset);
      const int32_t other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
      if (other_value > best_value || (other_value == best_value && other_index < best_index)) {
        best_value = other_value;
        best_index = other_index;
      }
    }
    if (lane == 0) {
      output[row] = best_index;
    }
  }
}

} // namespace argmax_kernel
} // namespace kernels
} // namespace omni_runtime

