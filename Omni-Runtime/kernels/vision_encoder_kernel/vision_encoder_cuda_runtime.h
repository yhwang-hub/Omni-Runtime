#pragma once

#include "kernels/operator_runtime_common.h"

#include <cmath>

#include "kernels/cuda_operator_runtime.h"

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
    bool is_fp16_projection = false;
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
    if (args.is_fp16_projection) {
      OMNI_RETURN_VAL_IF_LOG((!std::is_same_v<T, float>), false, ERROR,
                             "FP16 patch projection requires FP32 tensor storage");
      OMNI_RETURN_VAL_IF_LOG(!CudaKernelRuntime::HasTensorMma(), false, ERROR,
                             "FP16 patch projection requires CUDA tensor MMA");
      const int32_t patches =
          (args.height / args.spatial_patch) * (args.width / args.spatial_patch);
      return CudaKernelRuntime::CompileAndLaunch(
          tag,
          CudaKernelRuntime::MakeKernelSource<T, 1>(
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
    if constexpr (sizeof(T) == 2U && !std::is_integral_v<T>) {
      if (CudaKernelRuntime::HasTensorMma()) {
        const int32_t patches =
            (args.height / args.spatial_patch) * (args.width / args.spatial_patch);
        return CudaKernelRuntime::CompileAndLaunch(
            tag,
            CudaKernelRuntime::MakeKernelSource<T, 0>(
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
    bool is_ggml_interpolation = false;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(
        args.input == nullptr || args.position_weight == nullptr || args.output == nullptr, false);
    const std::string include = "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh";
    const std::string kernel =
        "omni_runtime::kernels::vision_encoder_kernel::VisionAddPositionImpl";
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        args.is_ggml_interpolation ? CudaKernelRuntime::MakeKernelSource<T, 1>(include, kernel)
                                   : CudaKernelRuntime::MakeKernelSource<T, 0>(include, kernel),
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
    bool is_ggml_gelu = false;
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
        args.element_count, args.row_size, args.is_approximate, args.is_ggml_gelu,
        static_cast<T *>(args.output));
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
    bool is_fp16_key_value = false;
    bool is_ggml_rope = false;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.qkv == nullptr || args.query == nullptr || args.key == nullptr ||
                           args.value == nullptr,
                       false);
    const std::string include = "kernels/vision_encoder_kernel/vision_encoder_kernel.cuh";
    const std::string kernel = "omni_runtime::kernels::vision_encoder_kernel::VisionQkvRopeImpl";
    const float theta_scale =
        args.is_ggml_rope ? std::pow(10000.0F, -2.0F / static_cast<float>(args.head_size / 2))
                          : 1.0F;
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        args.is_fp16_key_value ? CudaKernelRuntime::MakeKernelSource<T, 1>(include, kernel)
                               : CudaKernelRuntime::MakeKernelSource<T, 0>(include, kernel),
        args.options, static_cast<const T *>(args.qkv), args.token_count, args.grid_height,
        args.grid_width, args.merge_size, args.head_count, args.head_size, args.is_ggml_rope,
        theta_scale, static_cast<T *>(args.query), args.key, args.value);
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

template <typename T> struct CudaOperator<T, backend::PairedPatchEmbedding> final {
  static bool Launch(const backend::PairedPatchEmbedding &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.input.IsValid() || !args.first_weight.IsValid() ||
                               !args.second_weight.IsValid() || !args.bias.IsValid() ||
                               !args.output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.height <= 0 || args.width <= 0 || args.channels <= 0 ||
                               args.output_channels <= 0 || args.spatial_patch <= 0,
                           false, ERROR, "Invalid compute operation dimensions");
    return VisionPatchEmbedPairKernelRuntime<T>::CompileAndLaunch(
        "backend_PairedPatchEmbedding",
        {.options = {.grid_dim = {static_cast<uint32_t>(uint64_t(args.height / args.spatial_patch) *
                                                        (args.width / args.spatial_patch) *
                                                        args.output_channels),
                                  1U, 1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .input = CudaAddress<void>(args.input),
         .first_weight = CudaAddress<void>(args.first_weight),
         .second_weight = CudaAddress<void>(args.second_weight),
         .bias = CudaAddress<void>(args.bias),
         .height = args.height,
         .width = args.width,
         .channels = args.channels,
         .output_channels = args.output_channels,
         .spatial_patch = args.spatial_patch,
         .output = CudaAddress<void>(args.output),
         .is_fp16_projection = args.is_fp16_projection});
  }
};

template <typename T> struct CudaOperator<T, backend::GridPositionEmbedding> final {
  static bool Launch(const backend::GridPositionEmbedding &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.input.IsValid() || !args.position_weight.IsValid() ||
                               !args.output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.grid_height <= 0 || args.grid_width <= 0 ||
                               args.position_grid_size <= 0 || args.hidden_size <= 0 ||
                               args.merge_size <= 0,
                           false, ERROR, "Invalid compute operation dimensions");
    return VisionAddPositionKernelRuntime<T>::CompileAndLaunch(
        "backend_GridPositionEmbedding",
        {.options = {.grid_dim = {static_cast<uint32_t>((uint64_t(args.grid_height) *
                                                             args.grid_width * args.hidden_size +
                                                         255U) /
                                                        256U),
                                  1U, 1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .input = CudaAddress<void>(args.input),
         .position_weight = CudaAddress<void>(args.position_weight),
         .grid_height = args.grid_height,
         .grid_width = args.grid_width,
         .position_grid_size = args.position_grid_size,
         .hidden_size = args.hidden_size,
         .merge_size = args.merge_size,
         .output = CudaAddress<void>(args.output),
         .is_ggml_interpolation = args.is_ggml_interpolation});
  }
};

template <typename T> struct CudaOperator<T, backend::Bias> final {
  static bool Launch(const backend::Bias &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.input.IsValid() || !args.bias.IsValid() || !args.output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.row_size <= 0, false, ERROR,
                           "Invalid compute operation dimensions");
    return VisionBiasKernelRuntime<T>::CompileAndLaunch(
        "backend_Bias",
        {.options = {.grid_dim = {static_cast<uint32_t>((args.element_count + 255U) / 256U), 1U,
                                  1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .input = CudaAddress<void>(args.input),
         .bias = CudaAddress<void>(args.bias),
         .element_count = args.element_count,
         .row_size = args.row_size,
         .output = CudaAddress<void>(args.output)});
  }
};

template <typename T> struct CudaOperator<T, backend::BiasResidual> final {
  static bool Launch(const backend::BiasResidual &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.input.IsValid() || !args.bias.IsValid() ||
                               !args.residual.IsValid() || !args.output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.row_size <= 0, false, ERROR,
                           "Invalid compute operation dimensions");
    return VisionBiasResidualKernelRuntime<T>::CompileAndLaunch(
        "backend_BiasResidual",
        {.options = {.grid_dim = {static_cast<uint32_t>((args.element_count + 255U) / 256U), 1U,
                                  1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .input = CudaAddress<void>(args.input),
         .bias = CudaAddress<void>(args.bias),
         .residual = CudaAddress<void>(args.residual),
         .element_count = args.element_count,
         .row_size = args.row_size,
         .output = CudaAddress<void>(args.output)});
  }
};

template <typename T> struct CudaOperator<T, backend::GeluBias> final {
  static bool Launch(const backend::GeluBias &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.input.IsValid() || !args.bias.IsValid() || !args.output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.row_size <= 0, false, ERROR,
                           "Invalid compute operation dimensions");
    OMNI_RETURN_VAL_IF_LOG(args.is_ggml_gelu && !args.is_approximate, false, ERROR,
                           "GGML GELU requires the approximate activation");
    return VisionGeluBiasKernelRuntime<T>::CompileAndLaunch(
        "backend_GeluBias",
        {.options = {.grid_dim = {static_cast<uint32_t>((args.element_count + 255U) / 256U), 1U,
                                  1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .input = CudaAddress<void>(args.input),
         .bias = CudaAddress<void>(args.bias),
         .element_count = args.element_count,
         .row_size = args.row_size,
         .is_approximate = args.is_approximate,
         .is_ggml_gelu = args.is_ggml_gelu,
         .output = CudaAddress<void>(args.output)});
  }
};

template <typename T> struct CudaOperator<T, backend::GridQkvRope> final {
  static bool Launch(const backend::GridQkvRope &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.qkv.IsValid() || !args.query.IsValid() || !args.key.IsValid() ||
                               !args.value.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.grid_height <= 0 || args.grid_width <= 0 ||
                               args.merge_size <= 0 || args.head_count <= 0 ||
                               args.head_size <= 0 || args.head_size % 4 != 0 ||
                               int64_t(args.grid_height) * args.grid_width != args.token_count ||
                               args.grid_height % args.merge_size != 0 ||
                               args.grid_width % args.merge_size != 0 ||
                               (args.is_fp16_key_value && !std::is_same_v<T, float>),
                           false, ERROR, "Invalid compute operation dimensions");
    const uint64_t count = uint64_t(args.token_count) * args.head_count * args.head_size;
    const size_t key_value_bytes = args.is_fp16_key_value ? sizeof(uint16_t) : sizeof(T);
    OMNI_RETURN_VAL_IF_LOG(!HasElements(args.qkv, count * 3U, sizeof(T)) ||
                               !HasElements(args.query, count, sizeof(T)) ||
                               !HasElements(args.key, count, key_value_bytes) ||
                               !HasElements(args.value, count, key_value_bytes),
                           false, ERROR, "Invalid vision rotary buffers");
    return VisionQkvRopeKernelRuntime<T>::CompileAndLaunch(
        "backend_GridQkvRope",
        {.options = {.grid_dim = {static_cast<uint32_t>((uint64_t(args.token_count) *
                                                             args.head_count * args.head_size +
                                                         255U) /
                                                        256U),
                                  1U, 1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .qkv = CudaAddress<void>(args.qkv),
         .token_count = args.token_count,
         .grid_height = args.grid_height,
         .grid_width = args.grid_width,
         .merge_size = args.merge_size,
         .head_count = args.head_count,
         .head_size = args.head_size,
         .query = CudaAddress<void>(args.query),
         .key = CudaAddress<void>(args.key),
         .value = CudaAddress<void>(args.value),
         .is_fp16_key_value = args.is_fp16_key_value,
         .is_ggml_rope = args.is_ggml_rope});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
