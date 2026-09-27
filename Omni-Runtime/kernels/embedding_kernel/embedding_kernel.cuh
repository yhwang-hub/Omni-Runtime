#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace embedding_kernel {

template <typename T>
__device__ void EmbeddingImpl(const int32_t *const __restrict__ indices,
                              const T *const __restrict__ weight, const int32_t token_count,
                              const int32_t hidden_size, T *const __restrict__ output) {
  const int32_t token = static_cast<int32_t>(blockIdx.x);
  if (token >= token_count) {
    return;
  }
  const int64_t source_offset = static_cast<int64_t>(indices[token]) * hidden_size;
  const int64_t output_offset = static_cast<int64_t>(token) * hidden_size;
  for (int32_t dimension = static_cast<int32_t>(threadIdx.x); dimension < hidden_size;
       dimension += static_cast<int32_t>(blockDim.x)) {
    output[output_offset + dimension] = weight[source_offset + dimension];
  }
}

} // namespace embedding_kernel
} // namespace kernels
} // namespace omni_runtime

