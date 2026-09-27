#pragma once

#include <cstdint>
#include <string>
#include <algorithm>
#include <cmath>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class Qwen3VLPreprocessKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    uint64_t pixel_count;
    const void *const source;
    int32_t source_width_pixels;
    int32_t source_height_pixels;
    int32_t patch_size;
    void *const destination;
    int32_t input_width_pixels = 0;
    int32_t input_height_pixels = 0;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.source == nullptr || args.destination == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(args.source_width_pixels <= 0 || args.source_height_pixels <= 0 ||
                               args.patch_size <= 0 ||
                               args.source_width_pixels % args.patch_size != 0 ||
                               args.source_height_pixels % args.patch_size != 0 ||
                               args.pixel_count != static_cast<uint64_t>(args.source_width_pixels) *
                                                       args.source_height_pixels ||
                               args.input_width_pixels < 0 || args.input_height_pixels < 0 ||
                               ((args.input_width_pixels == 0) != (args.input_height_pixels == 0)),
                           false, ERROR, "Invalid Qwen3-VL preprocessing dimensions");
    if (args.input_width_pixels > 0 && args.input_height_pixels > 0) {
      return CudaKernelRuntime::CompileAndLaunch(
          tag,
          CudaKernelRuntime::MakeKernelSource<T>(
              "kernels/image_kernel/preprocess.cuh",
              "omni_runtime::inference::example::Qwen3VLResizePreprocessImpl"),
          args.options, static_cast<const uint8_t *>(args.source), args.input_width_pixels,
          args.input_height_pixels, args.source_width_pixels, args.source_height_pixels,
          args.patch_size, ResizePrecision(args.input_width_pixels, args.source_width_pixels),
          ResizePrecision(args.input_height_pixels, args.source_height_pixels),
          static_cast<T *>(args.destination));
    }
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/image_kernel/preprocess.cuh",
            "omni_runtime::inference::example::Qwen3VLPreprocessImpl"),
        args.options, args.pixel_count, static_cast<const uint8_t *>(args.source),
        args.source_width_pixels, args.source_height_pixels, args.patch_size,
        static_cast<T *>(args.destination));
  }

private:
  static int32_t ResizePrecision(const int32_t input_size, const int32_t output_size) {
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
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
