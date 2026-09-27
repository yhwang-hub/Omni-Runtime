#pragma once
#include <cstdint>
#include <cuda_fp16.h>
namespace omni_runtime {
namespace inference {
namespace example {
__device__ void Qwen3VLPreprocessImpl(const uint64_t pixel_count, const uint8_t *const source,
                                        const int32_t source_width_pixels,
                                        const int32_t source_height_pixels,
                                        const int32_t patch_size, uint16_t *const destination) {
  const uint64_t pixel_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel_index >= pixel_count) {
    return;
  }
  const int32_t row = static_cast<int32_t>(pixel_index / source_width_pixels);
  const int32_t column = static_cast<int32_t>(pixel_index % source_width_pixels);
  if (row >= source_height_pixels) {
    return;
  }
  const int32_t patch_area = patch_size * patch_size;
  const int32_t patch_values = 3 * patch_area;
  const int32_t patch_count_x = source_width_pixels / patch_size;
  const int32_t patch_index = (row / patch_size) * patch_count_x + column / patch_size;
  const int32_t local_index = (row % patch_size) * patch_size + column % patch_size;
  const uint8_t *const pixel =
      source + (static_cast<size_t>(row) * source_width_pixels + column) * 3U;
  const int64_t destination_offset = static_cast<int64_t>(patch_index) * patch_values + local_index;
#pragma unroll
  for (int32_t channel = 0; channel < 3; ++channel) {
    const __half value = __float2half_rn(static_cast<float>(pixel[2 - channel]) / 127.5f - 1.0f);
    destination[destination_offset + static_cast<int64_t>(channel) * patch_area] =
        *reinterpret_cast<const uint16_t *>(&value);
  }
}
} // namespace example
} // namespace inference
} // namespace omni_runtime
