#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace swiglu_kernel {

template <typename T>
__device__ void SwigluImpl(const T *const __restrict__ gate, const T *const __restrict__ up,
                           const uint64_t element_count, T *const __restrict__ output) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (; index < element_count; index += stride) {
    const float gate_value = ToFloat(gate[index]);
    const float silu = gate_value / (1.0f + expf(-gate_value));
    output[index] = FromFloat<T>(silu * ToFloat(up[index]));
  }
}

} // namespace swiglu_kernel
} // namespace kernels
} // namespace omni_runtime

