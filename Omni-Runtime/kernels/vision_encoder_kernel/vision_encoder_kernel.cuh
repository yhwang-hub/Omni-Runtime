#pragma once

#include <cstdint>
#include <type_traits>

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

__device__ inline float VisionGgmlGelu(const float value) {
  const float theta = 0.7978845608028654F * value * (1.0F + 0.044715F * value * value);
  float activation;
#if __CUDA_ARCH__ >= 750
  asm("tanh.approx.f32 %0, %1;" : "=f"(activation) : "f"(theta));
#else
  activation = tanhf(theta);
#endif
  return 0.5F * value * (1.0F + activation);
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

template <typename T, int32_t kIsGgmlInterpolation>
__device__ inline float
VisionPositionValue(const T *const position_embedding, const int32_t position_grid_size,
                    const int32_t hidden_size, const int32_t grid_height, const int32_t grid_width,
                    const int32_t patch_row, const int32_t patch_column, const int32_t dimension) {
  const float scale_y = __fdiv_rn(
      static_cast<float>(grid_height > 1 && position_grid_size > 1 ? grid_height - 1 : grid_height),
      static_cast<float>(grid_height > 1 && position_grid_size > 1 ? position_grid_size - 1
                                                                   : position_grid_size));
  const float scale_x = __fdiv_rn(
      static_cast<float>(grid_width > 1 && position_grid_size > 1 ? grid_width - 1 : grid_width),
      static_cast<float>(grid_width > 1 && position_grid_size > 1 ? position_grid_size - 1
                                                                  : position_grid_size));
  const float source_y = kIsGgmlInterpolation != 0 ? __fdividef(float(patch_row), scale_y)
                         : grid_height == 1        ? 0.0F
                                                   : static_cast<float>(patch_row) *
                                                  static_cast<float>(position_grid_size - 1) /
                                                  static_cast<float>(grid_height - 1);
  const float source_x = kIsGgmlInterpolation != 0 ? __fdividef(float(patch_column), scale_x)
                         : grid_width == 1         ? 0.0F
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
  if constexpr (kIsGgmlInterpolation != 0) {
    const float dx = fminf(fmaxf(x_fraction, 0.0F), 1.0F);
    const float dy = fminf(fmaxf(y_fraction, 0.0F), 1.0F);
    const float first = (ToFloat(position_embedding[offset00]) * (1.0F - dx)) * (1.0F - dy);
    const float second = fmaf(ToFloat(position_embedding[offset01]) * dx, 1.0F - dy, first);
    const float third = fmaf(ToFloat(position_embedding[offset10]) * (1.0F - dx), dy, second);
    return fmaf(ToFloat(position_embedding[offset11]) * dx, dy, third);
  }
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

template <typename T, int32_t kSeparateFrames>
__global__ void VisionPatchEmbedPairTensorImpl(const T *const input, const T *const first_weight,
                                               const T *const second_weight, const T *const bias,
                                               const int32_t patches, const int32_t inner,
                                               const int32_t columns, T *const output) {
  constexpr int32_t kTile = 64;
  constexpr int32_t kInner = 32;
  constexpr int32_t kStride = kInner + 8;
  using M = std::conditional_t<kSeparateFrames != 0, __half, T>;
  __shared__ M inputs[kTile * kStride];
  __shared__ M weights[2][kTile * kStride];
  const int32_t lane = threadIdx.x % 32;
  const int32_t warp = threadIdx.x / 32;
  const int32_t row = lane / 4;
  const int32_t column = (lane % 4) * 2;
  float accumulators[2][kTile / 8][4] = {};
  for (int32_t first = 0; first < inner; first += kInner) {
    for (int32_t index = threadIdx.x; index < kTile * kInner; index += blockDim.x) {
      const int32_t local_row = index / kInner;
      const int32_t local_column = index % kInner;
      const int32_t input_row = blockIdx.x * kTile + local_row;
      const int32_t weight_row = blockIdx.y * kTile + local_row;
      const int32_t d = first + local_column;
      const int32_t shared_offset = local_row * kStride + local_column;
      const int64_t weight_offset = static_cast<int64_t>(weight_row) * inner + d;
      if constexpr (kSeparateFrames != 0) {
        inputs[shared_offset] =
            input_row < patches && d < inner
                ? FromFloat<M>(ToFloat(input[static_cast<int64_t>(input_row) * inner + d]))
                : FromFloat<M>(0.0F);
        weights[0][shared_offset] = weight_row < columns && d < inner
                                        ? FromFloat<M>(ToFloat(first_weight[weight_offset]))
                                        : FromFloat<M>(0.0F);
        weights[1][shared_offset] = weight_row < columns && d < inner
                                        ? FromFloat<M>(ToFloat(second_weight[weight_offset]))
                                        : FromFloat<M>(0.0F);
      } else {
        inputs[shared_offset] = input_row < patches && d < inner
                                    ? input[static_cast<int64_t>(input_row) * inner + d]
                                    : FromFloat<T>(0.0F);
        weights[0][shared_offset] =
            weight_row < columns && d < inner ? first_weight[weight_offset] : FromFloat<T>(0.0F);
        weights[1][shared_offset] =
            weight_row < columns && d < inner ? second_weight[weight_offset] : FromFloat<T>(0.0F);
      }
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
          if constexpr (kSeparateFrames != 0) {
            float *const accumulator = accumulators[frame][tile];
            uint32_t c0 =
                PackPair(__float2half_rn(accumulator[0]), __float2half_rn(accumulator[1]));
            uint32_t c1 =
                PackPair(__float2half_rn(accumulator[2]), __float2half_rn(accumulator[3]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                         "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                         : "+r"(c0), "+r"(c1)
                         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            accumulator[0] = __half2float(__ushort_as_half(c0 & 65535U));
            accumulator[1] = __half2float(__ushort_as_half(c0 >> 16U));
            accumulator[2] = __half2float(__ushort_as_half(c1 & 65535U));
            accumulator[3] = __half2float(__ushort_as_half(c1 >> 16U));
          } else {
            TensorMma<M>(a, b, accumulators[0][tile]);
          }
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
        float sum = accumulators[0][tile][item];
        if constexpr (kSeparateFrames != 0) {
          sum = __half2float(__float2half_rn(sum)) +
                __half2float(__float2half_rn(accumulators[1][tile][item]));
        }
        output[static_cast<int64_t>(input_row) * columns + output_column] =
            FromFloat<T>(sum + ToFloat(bias[output_column]));
      }
    }
  }
}

template <typename T, int32_t kIsGgmlInterpolation>
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
    const float position = VisionPositionValue<T, kIsGgmlInterpolation>(
        position_embedding, position_grid_size, hidden_size, grid_height, grid_width, patch_row,
        patch_column, dimension);
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
                                   const bool is_ggml_gelu, T *const __restrict__ output) {
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < element_count; index += stride) {
    const float biased =
        ToFloat(FromFloat<T>(ToFloat(input[index]) + ToFloat(bias[index % row_size])));
    output[index] = FromFloat<T>(is_ggml_gelu ? VisionGgmlGelu(biased)
                                 : is_approximate
                                     ? VisionGelu(biased)
                                     : 0.5F * biased * (1.0F + erff(biased * 0.7071067811865475F)));
  }
}

template <typename T, int32_t kIsFp16KeyValue>
__global__ void VisionQkvRopeImpl(const T *const __restrict__ qkv, const int32_t token_count,
                                  const int32_t grid_height, const int32_t grid_width,
                                  const int32_t merge_size, const int32_t head_count,
                                  const int32_t head_size, const bool is_ggml_rope,
                                  const float theta_scale, T *const __restrict__ query,
                                  void *const __restrict__ key, void *const __restrict__ value) {
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
    const float angle = static_cast<float>(position) *
                        (is_ggml_rope ? __powf(theta_scale, static_cast<float>(frequency_index))
                                      : powf(10000.0F, exponent));
    const float cosine = is_ggml_rope ? __cosf(angle) : cosf(angle);
    const float sine = is_ggml_rope ? __sinf(angle) : sinf(angle);
    const int64_t token_offset = static_cast<int64_t>(token) * 3 * hidden_size;
    const int64_t source_offset = token_offset + head * head_size + dimension;
    const int64_t paired_offset = token_offset + head * head_size + paired_dimension;
    const float sign = dimension < half_size ? -1.0F : 1.0F;
    float rotated_query =
        ToFloat(qkv[source_offset]) * cosine + sign * ToFloat(qkv[paired_offset]) * sine;
    const int64_t key_offset = source_offset + hidden_size;
    const int64_t paired_key_offset = paired_offset + hidden_size;
    float rotated_key =
        ToFloat(qkv[key_offset]) * cosine + sign * ToFloat(qkv[paired_key_offset]) * sine;
    if (is_ggml_rope) {
      const int64_t first = token_offset + head * head_size + dimension % half_size;
      const float q0 = ToFloat(qkv[first]);
      const float q1 = ToFloat(qkv[first + half_size]);
      const float k0 = ToFloat(qkv[first + hidden_size]);
      const float k1 = ToFloat(qkv[first + hidden_size + half_size]);
      rotated_query =
          dimension < half_size ? fmaf(q0, cosine, -q1 * sine) : fmaf(q0, sine, q1 * cosine);
      rotated_key =
          dimension < half_size ? fmaf(k0, cosine, -k1 * sine) : fmaf(k0, sine, k1 * cosine);
    }
    query[index] = FromFloat<T>(rotated_query);
    if constexpr (kIsFp16KeyValue != 0) {
      static_cast<__half *>(key)[index] = __float2half_rn(rotated_key);
      static_cast<__half *>(value)[index] =
          __float2half_rn(ToFloat(qkv[source_offset + 2 * hidden_size]));
    } else {
      static_cast<T *>(key)[index] = FromFloat<T>(rotated_key);
      static_cast<T *>(value)[index] = qkv[source_offset + 2 * hidden_size];
    }
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
