#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace gated_add_kernel {

template <typename T>
__device__ void GatedAddImpl(const T *const __restrict__ lhs, const T *const __restrict__ rhs,
                             const T *const __restrict__ gate, const uint64_t element_count,
                             const int32_t hidden_size, T *const __restrict__ output) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (; index < element_count; index += stride) {
    const int32_t dimension = static_cast<int32_t>(index % hidden_size);
    output[index] =
        FromFloat<T>(ToFloat(lhs[index]) + ToFloat(rhs[index]) * ToFloat(gate[dimension]));
  }
}

} // namespace gated_add_kernel
} // namespace kernels
} // namespace omni_runtime

