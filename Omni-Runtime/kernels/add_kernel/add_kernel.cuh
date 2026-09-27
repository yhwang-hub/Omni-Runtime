#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace add_kernel {

template <typename T>
__device__ void AddImpl(const T *const __restrict__ lhs, const T *const __restrict__ rhs,
                        const uint64_t element_count, T *const __restrict__ output) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (; index < element_count; index += stride) {
    output[index] = FromFloat<T>(ToFloat(lhs[index]) + ToFloat(rhs[index]));
  }
}

} // namespace add_kernel
} // namespace kernels
} // namespace omni_runtime

