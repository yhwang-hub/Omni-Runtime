#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace vision_encoder_kernel {

template <typename T>
__device__ void
VisionPatchEmbedImpl(const T *const __restrict__ input, const T *const __restrict__ weight,
                     const T *const __restrict__ bias, const int32_t frames, const int32_t height,
                     const int32_t width, const int32_t channels, const int32_t output_channels,
                     const int32_t temporal_patch, const int32_t spatial_patch,
                     T *const __restrict__ output) {
  const int32_t output_frames = frames / temporal_patch;
  const int32_t output_height = height / spatial_patch;
  const int32_t output_width = width / spatial_patch;
  const int32_t patch_count = output_frames * output_height * output_width;
  const int32_t output_index = static_cast<int32_t>(blockIdx.x);
  if (output_index >= patch_count * output_channels) {
    return;
  }
  const int32_t patch_index = output_index / output_channels;
  const int32_t output_channel = output_index - patch_index * output_channels;
  const int32_t patch_x = patch_index % output_width;
  const int32_t patch_y = (patch_index / output_width) % output_height;
  const int32_t patch_t = patch_index / (output_height * output_width);
  const int32_t patch_volume = channels * temporal_patch * spatial_patch * spatial_patch;
  const int64_t weight_offset = static_cast<int64_t>(output_channel) * patch_volume;
  float partial_sum = 0.0f;
  for (int32_t index = static_cast<int32_t>(threadIdx.x); index < patch_volume;
       index += static_cast<int32_t>(blockDim.x)) {
    int32_t remainder = index;
    const int32_t patch_column = remainder % spatial_patch;
    remainder /= spatial_patch;
    const int32_t patch_row = remainder % spatial_patch;
    remainder /= spatial_patch;
    const int32_t patch_frame = remainder % temporal_patch;
    const int32_t channel = remainder / temporal_patch;
    const int32_t input_frame = patch_t * temporal_patch + patch_frame;
    const int32_t input_row = patch_y * spatial_patch + patch_row;
    const int32_t input_column = patch_x * spatial_patch + patch_column;
    const int64_t input_index =
        ((static_cast<int64_t>(channel) * frames + input_frame) * height + input_row) * width +
        input_column;
    partial_sum =
        fmaf(ToFloat(input[input_index]), ToFloat(weight[weight_offset + index]), partial_sum);
  }
  __shared__ float warp_values[32];
  partial_sum = BlockReduceSum(partial_sum, warp_values);
  if (threadIdx.x == 0) {
    output[output_index] = FromFloat<T>(partial_sum + ToFloat(bias[output_channel]));
  }
}

__device__ inline float VisionGelu(const float source) {
  constexpr float kScale = 0.7978845608028654f;
  constexpr float kCubicScale = 0.044715f;
  return 0.5f * source * (1.0f + tanhf(kScale * (source + kCubicScale * source * source * source)));
}

__device__ inline int32_t VisionMergedPatchIndex(const int32_t output_index,
                                                 const int32_t grid_height,
                                                 const int32_t grid_width,
                                                 const int32_t merge_size) {
  const int32_t local_area = merge_size * merge_size;
  const int32_t group_index = output_index / local_area;
  const int32_t local_index = output_index - group_index * local_area;
  const int32_t group_width = grid_width / merge_size;
  const int32_t group_row = group_index / group_width;
  const int32_t group_column = group_index - group_row * group_width;
  const int32_t local_row = local_index / merge_size;
  const int32_t local_column = local_index - local_row * merge_size;
  return (group_row * merge_size + local_row) * grid_width + group_column * merge_size +
         local_column;
}

__device__ inline float VisionPositionValue(const __half *const position_embedding,
                                            const int32_t position_grid_size,
                                            const int32_t hidden_size, const int32_t grid_height,
                                            const int32_t grid_width, const int32_t patch_row,
                                            const int32_t patch_column, const int32_t dimension) {
  const float source_y = grid_height == 1
                             ? 0.0f
                             : static_cast<float>(patch_row) * (position_grid_size - 1) /
                                   static_cast<float>(grid_height - 1);
  const float source_x = grid_width == 1
                             ? 0.0f
                             : static_cast<float>(patch_column) * (position_grid_size - 1) /
                                   static_cast<float>(grid_width - 1);
  const int32_t y0 = static_cast<int32_t>(floorf(source_y));
  const int32_t x0 = static_cast<int32_t>(floorf(source_x));
  const int32_t y1 = min(y0 + 1, position_grid_size - 1);
  const int32_t x1 = min(x0 + 1, position_grid_size - 1);
  const float y_fraction = source_y - y0;
  const float x_fraction = source_x - x0;
  const int64_t offset00 =
      (static_cast<int64_t>(y0) * position_grid_size + x0) * hidden_size + dimension;
  const int64_t offset01 =
      (static_cast<int64_t>(y0) * position_grid_size + x1) * hidden_size + dimension;
  const int64_t offset10 =
      (static_cast<int64_t>(y1) * position_grid_size + x0) * hidden_size + dimension;
  const int64_t offset11 =
      (static_cast<int64_t>(y1) * position_grid_size + x1) * hidden_size + dimension;
  const float top =
      fmaf(ToFloat(position_embedding[offset01]) - ToFloat(position_embedding[offset00]),
           x_fraction, ToFloat(position_embedding[offset00]));
  const float bottom =
      fmaf(ToFloat(position_embedding[offset11]) - ToFloat(position_embedding[offset10]),
           x_fraction, ToFloat(position_embedding[offset10]));
  return fmaf(bottom - top, y_fraction, top);
}

} // namespace vision_encoder_kernel
} // namespace kernels
} // namespace omni_runtime

