#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace flow_step_kernel {

// One explicit Euler integration step of the pi0.5 flow-matching sampler:
// x_next = x + dt * velocity. Applied elementwise over the action block.
template <typename T>
__device__ void FlowStepImpl(const T *const __restrict__ state,
                             const T *const __restrict__ velocity, const float dt,
                             const uint64_t element_count, T *const __restrict__ output) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (; index < element_count; index += stride) {
    const float next = ToFloat(state[index]) + dt * ToFloat(velocity[index]);
    output[index] = FromFloat<T>(next);
  }
}

} // namespace flow_step_kernel
} // namespace kernels
} // namespace omni_runtime

