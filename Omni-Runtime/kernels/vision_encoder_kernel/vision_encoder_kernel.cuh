#pragma once

#include <cstdint>

#include <kernels/common/kernel_type_utils.h>
#include <kernels/matmul_kernel/tensor_core.cuh>

namespace omni_runtime {
namespace kernels {
namespace vision_encoder_kernel {

__device__ inline float VisionGelu(const float value) {
  constexpr float kScale = 0.7978845608028654F;
  constexpr float kCubicScale = 0.044715F;
  return 0.5F * value * (1.0F + tanhf(kScale * (value + kCubicScale * value * value * value)));
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

template <typename T>
__device__ inline float
VisionPositionValue(const T *const position_embedding, const int32_t position_grid_size,
                    const int32_t hidden_size, const int32_t grid_height, const int32_t grid_width,
                    const int32_t patch_row, const int32_t patch_column, const int32_t dimension) {
  const float source_y = grid_height == 1 ? 0.0F
                                          : static_cast<float>(patch_row) *
                                                static_cast<float>(position_grid_size - 1) /
                                                static_cast<float>(grid_height - 1);
  const float source_x = grid_width == 1 ? 0.0F
                                         : static_cast<float>(patch_column) *
                                               static_cast<float>(position_grid_size - 1) /
                                               static_cast<float>(grid_width - 1);
  const int32_t y0 = static_cast<int32_t>(floorf(source_y));
  const int32_t x0 = static_cast<int32_t>(floorf(source_x));
  const int32_t y1 = min(y0 + 1, position_grid_size - 1);
  const int32_t x1 = min(x0 + 1, position_grid_size - 1);
  const float y_fraction = source_y - static_cast<float>(y0);
  const float x_fraction = source_x - static_cast<float>(x0);
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

template <typename T>
__global__ void
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
  float partial_sum = 0.0F;
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

template <typename T>
__global__ void VisionPatchEmbedPairImpl(const T *const __restrict__ input,
                                         const T *const __restrict__ first_weight,
                                         const T *const __restrict__ second_weight,
                                         const T *const __restrict__ bias, const int32_t height,
                                         const int32_t width, const int32_t channels,
                                         const int32_t output_channels, const int32_t spatial_patch,
                                         T *const __restrict__ output) {
  const int32_t grid_height = height / spatial_patch;
  const int32_t grid_width = width / spatial_patch;
  const int32_t output_index = static_cast<int32_t>(blockIdx.x);
  if (output_index >= grid_height * grid_width * output_channels) {
    return;
  }
  const int32_t patch_index = output_index / output_channels;
  const int32_t output_channel = output_index - patch_index * output_channels;
  const int32_t patch_column = patch_index % grid_width;
  const int32_t patch_row = patch_index / grid_width;
  const int32_t patch_area = spatial_patch * spatial_patch;
  const int32_t patch_values = channels * patch_area;
  const int64_t weight_offset = static_cast<int64_t>(output_channel) * patch_values;
  float partial_sum = 0.0F;
  for (int32_t index = static_cast<int32_t>(threadIdx.x); index < patch_values;
       index += static_cast<int32_t>(blockDim.x)) {
    const int32_t channel = index / patch_area;
    const int32_t patch_offset = index - channel * patch_area;
    const int32_t local_row = patch_offset / spatial_patch;
    const int32_t local_column = patch_offset - local_row * spatial_patch;
    const int64_t input_index =
        static_cast<int64_t>(patch_index) * patch_values + channel * patch_area + patch_offset;
    const float combined_weight = ToFloat(first_weight[weight_offset + index]) +
                                  ToFloat(second_weight[weight_offset + index]);
    partial_sum = fmaf(ToFloat(input[input_index]), combined_weight, partial_sum);
  }
  __shared__ float warp_values[32];
  partial_sum = BlockReduceSum(partial_sum, warp_values);
  if (threadIdx.x == 0) {
    output[output_index] = FromFloat<T>(partial_sum + ToFloat(bias[output_channel]));
  }
}

template <typename T>
__global__ void VisionPatchEmbedPairTensorImpl(const T *const input, const T *const first_weight,
                                               const T *const second_weight, const T *const bias,
                                               const int32_t patches, const int32_t inner,
                                               const int32_t columns, T *const output) {
  constexpr int32_t kTile = 64;
  constexpr int32_t kInner = 32;
  constexpr int32_t kStride = kInner + 8;
  __shared__ T inputs[kTile * kStride];
  __shared__ T weights[2][kTile * kStride];
  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  const int32_t row = lane / 4;
  const int32_t column = (lane % 4) * 2;
  float accumulators[kTile / 8][4] = {};
  for (int32_t first = 0; first < inner; first += kInner) {
    for (int32_t index = threadIdx.x; index < kTile * kInner; index += blockDim.x) {
      const int32_t local_row = index / kInner;
      const int32_t local_column = index % kInner;
      const int32_t input_row = blockIdx.x * kTile + local_row;
      const int32_t weight_row = blockIdx.y * kTile + local_row;
      const int32_t d = first + local_column;
      const int32_t shared_offset = local_row * kStride + local_column;
      inputs[shared_offset] = input_row < patches && d < inner
                                  ? input[static_cast<int64_t>(input_row) * inner + d]
                                  : FromFloat<T>(0.0F);
      const int64_t weight_offset = static_cast<int64_t>(weight_row) * inner + d;
      weights[0][shared_offset] =
          weight_row < columns && d < inner ? first_weight[weight_offset] : FromFloat<T>(0.0F);
      weights[1][shared_offset] =
          weight_row < columns && d < inner ? second_weight[weight_offset] : FromFloat<T>(0.0F);
    }
    __syncthreads();
#pragma unroll
    for (int32_t d = 0; d < kInner; d += 16) {
      uint32_t a[4];
#pragma unroll
      for (int32_t item = 0; item < 4; ++item) {
        a[item] = *reinterpret_cast<const uint32_t *>(
            inputs + (warp * 16 + row + (item % 2) * 8) * kStride + d + column + (item / 2) * 8);
      }
#pragma unroll
      for (int32_t tile = 0; tile < kTile / 8; ++tile) {
#pragma unroll
        for (int32_t frame = 0; frame < 2; ++frame) {
          const int32_t offset = (tile * 8 + row) * kStride + d + column;
          const uint32_t b[2] = {*reinterpret_cast<const uint32_t *>(weights[frame] + offset),
                                 *reinterpret_cast<const uint32_t *>(weights[frame] + offset + 8)};
          TensorMma<T>(a, b, accumulators[tile]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int32_t tile = 0; tile < kTile / 8; ++tile) {
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t input_row = blockIdx.x * kTile + warp * 16 + row + (item / 2) * 8;
      const int32_t output_column = blockIdx.y * kTile + tile * 8 + column + item % 2;
      if (input_row < patches && output_column < columns) {
        output[static_cast<int64_t>(input_row) * columns + output_column] =
            FromFloat<T>(accumulators[tile][item] + ToFloat(bias[output_column]));
      }
    }
  }
}

template <typename T>
__global__ void VisionAddPositionImpl(const T *const __restrict__ input,
                                      const T *const __restrict__ position_embedding,
                                      const int32_t grid_height, const int32_t grid_width,
                                      const int32_t position_grid_size, const int32_t hidden_size,
                                      const int32_t merge_size, T *const __restrict__ output) {
  const int64_t value_count = static_cast<int64_t>(grid_height) * grid_width * hidden_size;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < value_count; index += stride) {
    const int32_t dimension = static_cast<int32_t>(index % hidden_size);
    const int32_t output_patch = static_cast<int32_t>(index / hidden_size);
    const int32_t input_patch =
        VisionMergedPatchIndex(output_patch, grid_height, grid_width, merge_size);
    const int32_t patch_row = input_patch / grid_width;
    const int32_t patch_column = input_patch - patch_row * grid_width;
    const float position =
        VisionPositionValue(position_embedding, position_grid_size, hidden_size, grid_height,
                            grid_width, patch_row, patch_column, dimension);
    output[index] =
        FromFloat<T>(ToFloat(input[static_cast<int64_t>(input_patch) * hidden_size + dimension]) +
                     ToFloat(FromFloat<T>(position)));
  }
}

template <typename T>
__global__ void VisionBiasImpl(const T *const __restrict__ input, const T *const __restrict__ bias,
                               const uint64_t element_count, const int32_t row_size,
                               T *const __restrict__ output) {
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count; index += stride) {
    output[index] = FromFloat<T>(ToFloat(input[index]) + ToFloat(bias[index % row_size]));
  }
}

template <typename T>
__global__ void
VisionBiasResidualImpl(const T *const __restrict__ input, const T *const __restrict__ bias,
                       const T *const __restrict__ residual, const uint64_t element_count,
                       const int32_t row_size, T *const __restrict__ output) {
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count; index += stride) {
    const T biased = FromFloat<T>(ToFloat(input[index]) + ToFloat(bias[index % row_size]));
    output[index] = FromFloat<T>(ToFloat(biased) + ToFloat(residual[index]));
  }
}

template <typename T>
__global__ void VisionGeluBiasImpl(const T *const __restrict__ input,
                                   const T *const __restrict__ bias, const uint64_t element_count,
                                   const int32_t row_size, const bool is_approximate,
                                   T *const __restrict__ output) {
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count; index += stride) {
    const float biased =
        ToFloat(FromFloat<T>(ToFloat(input[index]) + ToFloat(bias[index % row_size])));
    output[index] =
        FromFloat<T>(is_approximate ? VisionGelu(biased)
                                    : 0.5F * biased * (1.0F + erff(biased * 0.7071067811865475F)));
  }
}

template <typename T>
__global__ void VisionQkvRopeImpl(const T *const __restrict__ qkv, const int32_t token_count,
                                  const int32_t grid_height, const int32_t grid_width,
                                  const int32_t merge_size, const int32_t head_count,
                                  const int32_t head_size, T *const __restrict__ query,
                                  T *const __restrict__ key, T *const __restrict__ value) {
  const int64_t value_count = static_cast<int64_t>(token_count) * head_count * head_size;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const int32_t hidden_size = head_count * head_size;
  const int32_t half_size = head_size / 2;
  const int32_t quarter_size = head_size / 4;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < value_count; index += stride) {
    const int32_t dimension = static_cast<int32_t>(index % head_size);
    const int64_t row = index / head_size;
    const int32_t token = static_cast<int32_t>(row / head_count);
    const int32_t head = static_cast<int32_t>(row % head_count);
    const int32_t paired_dimension =
        dimension < half_size ? dimension + half_size : dimension - half_size;
    const int32_t frequency_dimension = dimension % half_size;
    const int32_t frequency_index = frequency_dimension % quarter_size;
    const int32_t merged_width = grid_width / merge_size;
    const int32_t group_index = token / (merge_size * merge_size);
    const int32_t local_index = token - group_index * merge_size * merge_size;
    const int32_t local_row = local_index / merge_size;
    const int32_t local_column = local_index - local_row * merge_size;
    const int32_t patch_row = (group_index / merged_width) * merge_size + local_row;
    const int32_t patch_column = (group_index % merged_width) * merge_size + local_column;
    const int32_t position = frequency_dimension < quarter_size ? patch_row : patch_column;
    const float exponent =
        -2.0F * static_cast<float>(frequency_index) / static_cast<float>(half_size);
    const float angle = static_cast<float>(position) * powf(10000.0F, exponent);
    const float cosine = cosf(angle);
    const float sine = sinf(angle);
    const int64_t token_offset = static_cast<int64_t>(token) * 3 * hidden_size;
    const int64_t source_offset = token_offset + head * head_size + dimension;
    const int64_t paired_offset = token_offset + head * head_size + paired_dimension;
    const float sign = dimension < half_size ? -1.0F : 1.0F;
    query[index] = FromFloat<T>(ToFloat(qkv[source_offset]) * cosine +
                                sign * ToFloat(qkv[paired_offset]) * sine);
    const int64_t key_offset = source_offset + hidden_size;
    const int64_t paired_key_offset = paired_offset + hidden_size;
    key[index] = FromFloat<T>(ToFloat(qkv[key_offset]) * cosine +
                              sign * ToFloat(qkv[paired_key_offset]) * sine);
    value[index] = qkv[source_offset + 2 * hidden_size];
  }
}

template <typename T>
__global__ void VisionMergeImpl(const T *const __restrict__ input, const int32_t token_count,
                                const int32_t grid_width, const int32_t hidden_size,
                                const int32_t merge_size, T *const __restrict__ output) {
  const int32_t merged_token_count = token_count / (merge_size * merge_size);
  const int64_t value_count =
      static_cast<int64_t>(merged_token_count) * hidden_size * merge_size * merge_size;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < value_count; index += stride) {
    output[index] = input[index];
  }
}

} // namespace vision_encoder_kernel
} // namespace kernels
} // namespace omni_runtime
