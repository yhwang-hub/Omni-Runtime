#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace gated_attention_kernel {

template <typename T>
__device__ void SplitGatedQueryImpl(const T *const __restrict__ query_gate,
                                    const int32_t token_count, const int32_t head_count,
                                    const int32_t head_size, T *const __restrict__ query,
                                    T *const __restrict__ gate) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t element_count = static_cast<int64_t>(token_count) * head_count * head_size;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; index < element_count; index += stride) {
    const int32_t dimension = static_cast<int32_t>(index % head_size);
    const int64_t head_row = index / head_size;
    const int64_t source = head_row * 2 * head_size + dimension;
    query[index] = query_gate[source];
    gate[index] = query_gate[source + head_size];
  }
}

template <typename T>
__device__ void SigmoidGateImpl(const T *const __restrict__ input, const T *const __restrict__ gate,
                                const uint64_t element_count, T *const __restrict__ output) {
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (; index < element_count; index += stride) {
    const float gate_value = ToFloat(gate[index]);
    output[index] = FromFloat<T>(ToFloat(input[index]) / (1.0f + expf(-gate_value)));
  }
}

} // namespace gated_attention_kernel
} // namespace kernels
} // namespace omni_runtime

