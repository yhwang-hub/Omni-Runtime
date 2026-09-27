#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace mrope_kernel {

__device__ int32_t PositionAxis(const int32_t pair_index, const int32_t temporal_section,
                                const int32_t height_section) {
  if (pair_index < temporal_section) {
    return 0;
  }
  return pair_index < temporal_section + height_section ? 1 : 2;
}

template <typename T>
__device__ void RotateElements(const T *const __restrict__ input, const int32_t *const position_ids,
                               const int64_t element_count, const int32_t tokens,
                               const int32_t heads, const int32_t head_dim,
                               const int32_t temporal_section, const int32_t height_section,
                               const float frequency_base, T *const __restrict__ output) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int32_t half_dim = head_dim / 2;
  for (; index < element_count; index += stride) {
    const int32_t dimension = static_cast<int32_t>(index % head_dim);
    const int32_t token = static_cast<int32_t>(index / (static_cast<int64_t>(heads) * head_dim));
    const int32_t pair_index = dimension % half_dim;
    const int32_t position_axis = PositionAxis(pair_index, temporal_section, height_section);
    const float position =
        static_cast<float>(position_ids[static_cast<int64_t>(position_axis) * tokens + token]);
    const float exponent = -2.0f * static_cast<float>(pair_index) / static_cast<float>(head_dim);
    const float angle = position * powf(frequency_base, exponent);
    const int32_t rotated_dimension =
        dimension < half_dim ? dimension + half_dim : dimension - half_dim;
    const float rotated_sign = dimension < half_dim ? -1.0f : 1.0f;
    const int64_t rotated_index = index - dimension + rotated_dimension;
    output[index] = FromFloat<T>(ToFloat(input[index]) * cosf(angle) +
                                 rotated_sign * ToFloat(input[rotated_index]) * sinf(angle));
  }
}

template <typename T>
__device__ void MropeImpl(const T *const __restrict__ query, const T *const __restrict__ key,
                          const int32_t *const __restrict__ position_ids, const int32_t tokens,
                          const int32_t query_heads, const int32_t key_value_heads,
                          const int32_t head_dim, const int32_t temporal_section,
                          const int32_t height_section, const int32_t width_section,
                          const float frequency_base, T *const __restrict__ query_output,
                          T *const __restrict__ key_output) {
  if (head_dim <= 0 || head_dim % 2 != 0 || temporal_section < 0 || height_section < 0 ||
      width_section < 0 || temporal_section + height_section + width_section != head_dim / 2 ||
      frequency_base <= 0.0f) {
    return;
  }
  const int64_t query_element_count = static_cast<int64_t>(tokens) * query_heads * head_dim;
  const int64_t key_element_count = static_cast<int64_t>(tokens) * key_value_heads * head_dim;
  RotateElements(query, position_ids, query_element_count, tokens, query_heads, head_dim,
                 temporal_section, height_section, frequency_base, query_output);
  RotateElements(key, position_ids, key_element_count, tokens, key_value_heads, head_dim,
                 temporal_section, height_section, frequency_base, key_output);
}

} // namespace mrope_kernel
} // namespace kernels
} // namespace omni_runtime

