#pragma once

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class VisionPatchEmbedKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const weight;
    const void *const bias;
    int32_t frames;
    int32_t height;
    int32_t width;
    int32_t channels;
    int32_t output_channels;
    int32_t temporal_patch;
    int32_t spatial_patch;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.weight == nullptr || args.bias == nullptr ||
                           args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh",
            "omni_runtime::kernels::vision_encoder_kernel::VisionPatchEmbedImpl"),
        args.options, static_cast<const T *>(args.input), static_cast<const T *>(args.weight),
        static_cast<const T *>(args.bias), args.frames, args.height, args.width, args.channels,
        args.output_channels, args.temporal_patch, args.spatial_patch,
        static_cast<T *>(args.output));
  }
};

template <typename T> class VisionPatchEmbedPairKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const first_weight;
    const void *const second_weight;
    const void *const bias;
    int32_t height;
    int32_t width;
    int32_t channels;
    int32_t output_channels;
    int32_t spatial_patch;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.first_weight == nullptr ||
                           args.second_weight == nullptr || args.bias == nullptr ||
                           args.output == nullptr,
                       false);
    OMNI_RETURN_VAL_IF_LOG(args.height <= 0 || args.width <= 0 || args.channels <= 0 ||
                               args.output_channels <= 0 || args.spatial_patch <= 0 ||
                               args.height % args.spatial_patch != 0 ||
                               args.width % args.spatial_patch != 0,
                           false, ERROR, "Invalid paired vision patch dimensions");
    if constexpr (sizeof(T) == 2U && !std::is_integral_v<T>) {
      if (CudaKernelRuntime::HasTensorMma()) {
        const int32_t patches =
            (args.height / args.spatial_patch) * (args.width / args.spatial_patch);
        return CudaKernelRuntime::CompileAndLaunch(
            tag,
            CudaKernelRuntime::MakeKernelSource<T>(
                "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh",
                "omni_runtime::kernels::vision_encoder_kernel::VisionPatchEmbedPairTensorImpl"),
            {.grid_dim = {static_cast<uint32_t>((patches + 63) / 64),
                          static_cast<uint32_t>((args.output_channels + 63) / 64), 1U},
             .block_dim = {128U, 1U, 1U},
             .shared_memory_bytes = 0},
            static_cast<const T *>(args.input), static_cast<const T *>(args.first_weight),
            static_cast<const T *>(args.second_weight), static_cast<const T *>(args.bias), patches,
            args.channels * args.spatial_patch * args.spatial_patch, args.output_channels,
            static_cast<T *>(args.output));
      }
    }
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh",
            "omni_runtime::kernels::vision_encoder_kernel::VisionPatchEmbedPairImpl"),
        args.options, static_cast<const T *>(args.input), static_cast<const T *>(args.first_weight),
        static_cast<const T *>(args.second_weight), static_cast<const T *>(args.bias), args.height,
        args.width, args.channels, args.output_channels, args.spatial_patch,
        static_cast<T *>(args.output));
  }
};

template <typename T> class VisionAddPositionKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const position_weight;
    int32_t grid_height;
    int32_t grid_width;
    int32_t position_grid_size;
    int32_t hidden_size;
    int32_t merge_size;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(
        args.input == nullptr || args.position_weight == nullptr || args.output == nullptr, false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh",
            "omni_runtime::kernels::vision_encoder_kernel::VisionAddPositionImpl"),
        args.options, static_cast<const T *>(args.input),
        static_cast<const T *>(args.position_weight), args.grid_height, args.grid_width,
        args.position_grid_size, args.hidden_size, args.merge_size, static_cast<T *>(args.output));
  }
};

template <typename T> class VisionBiasKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const bias;
    uint64_t element_count;
    int32_t row_size;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.bias == nullptr || args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh",
            "omni_runtime::kernels::vision_encoder_kernel::VisionBiasImpl"),
        args.options, static_cast<const T *>(args.input), static_cast<const T *>(args.bias),
        args.element_count, args.row_size, static_cast<T *>(args.output));
  }
};

template <typename T> class VisionBiasResidualKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const bias;
    const void *const residual;
    uint64_t element_count;
    int32_t row_size;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.bias == nullptr || args.residual == nullptr ||
                           args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh",
            "omni_runtime::kernels::vision_encoder_kernel::VisionBiasResidualImpl"),
        args.options, static_cast<const T *>(args.input), static_cast<const T *>(args.bias),
        static_cast<const T *>(args.residual), args.element_count, args.row_size,
        static_cast<T *>(args.output));
  }
};

template <typename T> class VisionGeluBiasKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const bias;
    uint64_t element_count;
    int32_t row_size;
    bool is_approximate = true;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.bias == nullptr || args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh",
            "omni_runtime::kernels::vision_encoder_kernel::VisionGeluBiasImpl"),
        args.options, static_cast<const T *>(args.input), static_cast<const T *>(args.bias),
        args.element_count, args.row_size, args.is_approximate, static_cast<T *>(args.output));
  }
};

template <typename T> class VisionQkvRopeKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const qkv;
    int32_t token_count;
    int32_t grid_height;
    int32_t grid_width;
    int32_t merge_size;
    int32_t head_count;
    int32_t head_size;
    void *const query;
    void *const key;
    void *const value;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.qkv == nullptr || args.query == nullptr || args.key == nullptr ||
                           args.value == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh",
            "omni_runtime::kernels::vision_encoder_kernel::VisionQkvRopeImpl"),
        args.options, static_cast<const T *>(args.qkv), args.token_count, args.grid_height,
        args.grid_width, args.merge_size, args.head_count, args.head_size,
        static_cast<T *>(args.query), static_cast<T *>(args.key), static_cast<T *>(args.value));
  }
};

template <typename T> class VisionMergeKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    int32_t token_count;
    int32_t grid_width;
    int32_t hidden_size;
    int32_t merge_size;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.output == nullptr, false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh",
            "omni_runtime::kernels::vision_encoder_kernel::VisionMergeImpl"),
        args.options, static_cast<const T *>(args.input), args.token_count, args.grid_width,
        args.hidden_size, args.merge_size, static_cast<T *>(args.output));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
