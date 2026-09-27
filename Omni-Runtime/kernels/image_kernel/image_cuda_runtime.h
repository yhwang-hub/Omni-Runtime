#pragma once

#include "kernels/cuda_operator_runtime.h"
#include "kernels/image_kernel/image_resize.h"

#include <cstdint>
#include <string>
#include <algorithm>
#include <cmath>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class ImagePatchPreprocessKernelRuntime final {
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
    bool is_letterbox = false;
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
                               (args.is_letterbox && args.input_width_pixels == 0) ||
                               ((args.input_width_pixels == 0) != (args.input_height_pixels == 0)),
                           false, ERROR, "Invalid image patch preprocessing dimensions");
    if (args.input_width_pixels > 0 && args.input_height_pixels > 0) {
      const auto content =
          ResizeContentSize(args.input_width_pixels, args.input_height_pixels,
                            args.source_width_pixels, args.source_height_pixels, args.is_letterbox);
      return CudaKernelRuntime::CompileAndLaunch(
          tag,
          CudaKernelRuntime::MakeKernelSource<T>(
              "kernels/image_kernel/preprocess.cuh",
              "omni_runtime::kernels::image_kernel::ResizePatchPreprocessImpl"),
          args.options, static_cast<const uint8_t *>(args.source), args.input_width_pixels,
          args.input_height_pixels, args.source_width_pixels, args.source_height_pixels, content[0],
          content[1], args.patch_size, ResizePrecision(args.input_width_pixels, content[0]),
          ResizePrecision(args.input_height_pixels, content[1]),
          static_cast<T *>(args.destination));
    }
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/image_kernel/preprocess.cuh",
            "omni_runtime::kernels::image_kernel::ImagePatchPreprocessImpl"),
        args.options, args.pixel_count, static_cast<const uint8_t *>(args.source),
        args.source_width_pixels, args.source_height_pixels, args.patch_size,
        static_cast<T *>(args.destination));
  }
};

template <typename T> struct CudaOperator<T, backend::ImagePatchPreprocess> final {
  static bool Launch(const backend::ImagePatchPreprocess &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.source.IsValid() || !args.destination.IsValid(), false, ERROR,
                           "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.source_width_pixels <= 0 || args.source_height_pixels <= 0 ||
                               args.patch_size <= 0,
                           false, ERROR, "Invalid compute operation dimensions");
    return ImagePatchPreprocessKernelRuntime<T>::CompileAndLaunch(
        "backend_ImagePatchPreprocess",
        {.options = {.grid_dim = {static_cast<uint32_t>((args.pixel_count + 255U) / 256U), 1U, 1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .pixel_count = args.pixel_count,
         .source = CudaAddress<void>(args.source),
         .source_width_pixels = args.source_width_pixels,
         .source_height_pixels = args.source_height_pixels,
         .patch_size = args.patch_size,
         .destination = CudaAddress<void>(args.destination),
         .input_width_pixels = args.input_width_pixels,
         .input_height_pixels = args.input_height_pixels,
         .is_letterbox = args.is_letterbox});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
