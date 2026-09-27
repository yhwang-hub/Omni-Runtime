#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace omni_runtime::kernels::runtime {

inline std::array<int32_t, 2U>
ResizeContentSize(const int32_t input_width, const int32_t input_height, const int32_t output_width,
                  const int32_t output_height, const bool is_letterbox) {
  if (!is_letterbox) {
    return {output_width, output_height};
  }
  const float scale = std::min(static_cast<float>(output_width) / input_width,
                               static_cast<float>(output_height) / input_height);
  return {std::min(output_width, static_cast<int32_t>(std::ceil(input_width * scale))),
          std::min(output_height, static_cast<int32_t>(std::ceil(input_height * scale)))};
}

inline int32_t ResizePrecision(const int32_t input_size, const int32_t output_size) {
  const double scale = static_cast<double>(input_size) / output_size;
  const double filter_scale = std::max(scale, 1.0);
  const auto cubic = [](const double value) {
    const double x = std::fabs(value);
    return x < 1.0 ? ((1.5 * x - 2.5) * x) * x + 1.0
                   : (x < 2.0 ? ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0 : 0.0);
  };
  double maximum = 0.0;
  for (int32_t pixel = 0; pixel < output_size; ++pixel) {
    const double center = (pixel + 0.5) * scale;
    const int32_t begin = std::max(0, static_cast<int32_t>(center - 2.0 * filter_scale + 0.5));
    const int32_t end =
        std::min(input_size, static_cast<int32_t>(center + 2.0 * filter_scale + 0.5));
    double sum = 0.0;
    double peak = 0.0;
    for (int32_t index = begin; index < end; ++index) {
      const double weight = cubic((index - center + 0.5) / filter_scale);
      sum += weight;
      peak = std::max(peak, weight);
    }
    maximum = std::max(maximum, peak / sum);
  }
  int32_t precision = 0;
  while (precision < 21 && std::round(maximum * (1 << (precision + 1))) <= 32767.0) {
    ++precision;
  }
  return precision;
}

} // namespace omni_runtime::kernels::runtime
