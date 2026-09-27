#pragma once

#include "kernels/operator_runtime.h"
#include "kernels/image_kernel/image_resize.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::ImagePatchPreprocess> final {
  static bool Make(const backend::ImagePatchPreprocess &args,
                   backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const int32_t width =
        args.input_width_pixels > 0 ? args.input_width_pixels : args.source_width_pixels;
    const int32_t height =
        args.input_height_pixels > 0 ? args.input_height_pixels : args.source_height_pixels;
    OMNI_RETURN_VAL_IF_LOG(
        args.source_width_pixels <= 0 || args.source_height_pixels <= 0 || args.patch_size <= 0 ||
            width <= 0 || height <= 0 ||
            args.pixel_count != uint64_t(args.source_width_pixels) * args.source_height_pixels ||
            args.source_width_pixels % args.patch_size != 0 ||
            args.source_height_pixels % args.patch_size != 0 || args.input_width_pixels < 0 ||
            args.input_height_pixels < 0 ||
            ((args.input_width_pixels == 0) != (args.input_height_pixels == 0)) ||
            !HasElements(args.source, uint64_t(width) * height * 3U, sizeof(uint8_t)) ||
            !HasElements(args.destination, args.pixel_count * 3U, ElementBytes(call->type)),
        false, ERROR, "Invalid image preprocessing operation");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("image_kernel/image_kernel.cl", "ImagePatchPreprocess",
                                        (args.pixel_count + 255U) / 256U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.pixel_count, args.source, args.source_width_pixels,
                        args.source_height_pixels, args.patch_size, args.destination,
                        args.input_width_pixels, args.input_height_pixels,
                        ResizePrecision(width, args.source_width_pixels),
                        ResizePrecision(height, args.source_height_pixels));
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
