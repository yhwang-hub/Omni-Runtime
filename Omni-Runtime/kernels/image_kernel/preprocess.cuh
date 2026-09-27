#pragma once

#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace image_kernel {

__device__ inline double CubicWeight(const double distance) {
  const double x = fabs(distance);
  if (x < 1.0) {
    return ((1.5 * x - 2.5) * x) * x + 1.0;
  }
  return x < 2.0 ? ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0 : 0.0;
}

__device__ inline int32_t QuantizedWeight(const double weight, const int32_t precision) {
  const double value = weight * (1 << precision);
  return static_cast<int32_t>(value < 0.0 ? ceil(value - 0.5) : floor(value + 0.5));
}

__device__ inline uint8_t HorizontalPixel(const uint8_t *const source, const int32_t row,
                                          const int32_t column, const int32_t channel,
                                          const int32_t input_width, const int32_t output_width,
                                          const int32_t precision) {
  if (input_width == output_width) {
    return source[(static_cast<int64_t>(row) * input_width + column) * 3 + channel];
  }
  const double scale = static_cast<double>(input_width) / output_width;
  const double filter_scale = fmax(scale, 1.0);
  const double center = (column + 0.5) * scale;
  const int32_t begin = max(0, static_cast<int32_t>(center - 2.0 * filter_scale + 0.5));
  const int32_t end = min(input_width, static_cast<int32_t>(center + 2.0 * filter_scale + 0.5));
  double sum = 0.0;
  for (int32_t index = begin; index < end; ++index) {
    sum += CubicWeight((index - center + 0.5) / filter_scale);
  }
  int32_t value = 1 << (precision - 1);
  for (int32_t index = begin; index < end; ++index) {
    const int32_t weight =
        QuantizedWeight(CubicWeight((index - center + 0.5) / filter_scale) / sum, precision);
    value += weight * source[(static_cast<int64_t>(row) * input_width + index) * 3 + channel];
  }
  return static_cast<uint8_t>(min(255, max(0, value >> precision)));
}

template <typename T>
__global__ void ResizePatchPreprocessImpl(const uint8_t *const source, const int32_t input_width,
                                          const int32_t input_height, const int32_t output_width,
                                          const int32_t output_height, const int32_t content_width,
                                          const int32_t content_height, const int32_t patch_size,
                                          const int32_t x_precision, const int32_t y_precision,
                                          T *const destination) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= output_width * output_height) {
    return;
  }
  const int32_t output_row = index / output_width;
  const int32_t output_column = index % output_width;
  const int32_t patch_area = patch_size * patch_size;
  const int32_t patch =
      (output_row / patch_size) * (output_width / patch_size) + output_column / patch_size;
  const int32_t local = (output_row % patch_size) * patch_size + output_column % patch_size;
  const int32_t row = output_row - (output_height - content_height) / 2;
  const int32_t column = output_column - (output_width - content_width) / 2;
  if (row < 0 || row >= content_height || column < 0 || column >= content_width) {
    for (int32_t channel = 0; channel < 3; ++channel) {
      destination[static_cast<int64_t>(patch) * 3 * patch_area + channel * patch_area + local] =
          ::omni_runtime::kernels::FromFloat<T>(-1.0F);
    }
    return;
  }
  const double scale = static_cast<double>(input_height) / content_height;
  const double filter_scale = fmax(scale, 1.0);
  const double center = (row + 0.5) * scale;
  const int32_t begin = max(0, static_cast<int32_t>(center - 2.0 * filter_scale + 0.5));
  const int32_t end = min(input_height, static_cast<int32_t>(center + 2.0 * filter_scale + 0.5));
  double weight_sum = 0.0;
  for (int32_t y = begin; y < end; ++y) {
    weight_sum += CubicWeight((y - center + 0.5) / filter_scale);
  }
  for (int32_t channel = 0; channel < 3; ++channel) {
    int32_t sum = 1 << (y_precision - 1);
    for (int32_t y = begin; y < end; ++y) {
      const int32_t weight =
          QuantizedWeight(CubicWeight((y - center + 0.5) / filter_scale) / weight_sum, y_precision);
      sum += weight *
             HorizontalPixel(source, y, column, channel, input_width, content_width, x_precision);
    }
    const int32_t value = min(255, max(0, sum >> y_precision));
    destination[static_cast<int64_t>(patch) * 3 * patch_area + channel * patch_area + local] =
        ::omni_runtime::kernels::FromFloat<T>((static_cast<float>(value) - 127.5F) / 127.5F);
  }
}

template <typename T>
__global__ void ImagePatchPreprocessImpl(const uint64_t pixel_count, const uint8_t *const source,
                                         const int32_t source_width_pixels,
                                         const int32_t source_height_pixels,
                                         const int32_t patch_size, T *const destination) {
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
    const float value = static_cast<float>(pixel[channel]) / 127.5F - 1.0F;
    destination[destination_offset + static_cast<int64_t>(channel) * patch_area] =
        ::omni_runtime::kernels::FromFloat<T>(value);
  }
}

} // namespace image_kernel
} // namespace kernels
} // namespace omni_runtime
