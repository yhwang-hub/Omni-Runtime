#pragma once

#include "kernels/operator_runtime_common.h"

#include <cmath>

#include "kernels/opencl_operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::PairedPatchEmbedding> final {
  static bool Make(const backend::PairedPatchEmbedding &args,
                   backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(args.height <= 0 || args.width <= 0 || args.channels <= 0 ||
                               args.output_channels <= 0 || args.spatial_patch <= 0 ||
                               args.height % args.spatial_patch != 0 ||
                               args.width % args.spatial_patch != 0,
                           false, ERROR, "Invalid patch embedding dimensions");
    const size_t bytes = ElementBytes(call->type);
    const uint64_t patches =
        uint64_t(args.height / args.spatial_patch) * (args.width / args.spatial_patch);
    const uint64_t inner = uint64_t(args.channels) * args.spatial_patch * args.spatial_patch;
    OMNI_RETURN_VAL_IF_LOG(
        !HasElements(args.input, patches * inner, bytes) ||
            !HasElements(args.first_weight, uint64_t(args.output_channels) * inner, bytes) ||
            !HasElements(args.second_weight, uint64_t(args.output_channels) * inner, bytes) ||
            !HasElements(args.bias, args.output_channels, bytes) ||
            !HasElements(args.output, patches * args.output_channels, bytes),
        false, ERROR, "Invalid patch embedding buffers");
    OMNI_RETURN_VAL_IF_LOG(args.is_fp16_projection && call->type != backend::ScalarType::kFloat32,
                           false, ERROR, "FP16 patch projection requires FP32 tensor storage");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("vision_encoder_kernel/vision_encoder_kernel.cl",
                                        "PairedPatchEmbedding", patches * args.output_channels,
                                        256U, call),
                       false);
    if (args.is_fp16_projection) {
      OMNI_RETURN_VAL_IF_LOG(!call->has_nvidia_mma, false, ERROR,
                             "FP16 patch projection requires NVIDIA tensor MMA");
      call->build_defines.push_back("OMNI_PATCH_HALF_PROJECTION");
      call->build_defines.push_back("OMNI_HALF_ATTENTION");
    }
    if (call->has_nvidia_mma &&
        (call->type == backend::ScalarType::kFloat16 ||
         call->type == backend::ScalarType::kBFloat16 || args.is_fp16_projection)) {
      call->entry = "PairedPatchEmbeddingTensor";
      call->dimensions = 2U;
      call->local_size = {128U, 1U, 1U};
      call->global_size = {RoundWorkSize(patches, 64U) / 64U * 128U,
                           RoundWorkSize(args.output_channels, 64U) / 64U, 1U};
    }
    BindOpenCLArguments(call, args.input, args.first_weight, args.second_weight, args.bias,
                        args.height, args.width, args.channels, args.output_channels,
                        args.spatial_patch, args.output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::GridPositionEmbedding> final {
  static bool Make(const backend::GridPositionEmbedding &args,
                   backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(args.grid_height <= 0 || args.grid_width <= 0 ||
                               args.position_grid_size <= 0 || args.hidden_size <= 0 ||
                               args.merge_size <= 0 || args.grid_height % args.merge_size != 0 ||
                               args.grid_width % args.merge_size != 0,
                           false, ERROR, "Invalid position embedding dimensions");
    const uint64_t count = uint64_t(args.grid_height) * args.grid_width * args.hidden_size;
    const size_t bytes = ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(!HasElements(args.input, count, bytes) ||
                               !HasElements(args.output, count, bytes) ||
                               !HasElements(args.position_weight,
                                            uint64_t(args.position_grid_size) *
                                                args.position_grid_size * args.hidden_size,
                                            bytes),
                           false, ERROR, "Invalid position embedding buffers");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("vision_encoder_kernel/vision_encoder_kernel.cl",
                                        "GridPositionEmbedding", (count + 255U) / 256U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.input, args.position_weight, args.grid_height, args.grid_width,
                        args.position_grid_size, args.hidden_size, args.merge_size, args.output);
    if (args.is_ggml_interpolation) {
      call->build_defines.push_back("OMNI_GGML_INTERPOLATION");
    }
    return true;
  }
};

inline bool ConfigureBias(const std::string &entry, const backend::DeviceView &input,
                          const backend::DeviceView &bias, const uint64_t count,
                          const int32_t row_size, const backend::DeviceView &output,
                          backend::OpenCLInvocation *const call) {
  OMNI_RETURN_VAL_IF(call == nullptr, false);
  const size_t bytes = ElementBytes(call->type);
  OMNI_RETURN_VAL_IF_LOG(row_size <= 0 || count == 0U || !HasElements(input, count, bytes) ||
                             !HasElements(bias, row_size, bytes) ||
                             !HasElements(output, count, bytes),
                         false, ERROR, "Invalid bias operation");
  return ConfigureOpenCL("vision_encoder_kernel/vision_encoder_kernel.cl", entry,
                         (count + 255U) / 256U, 256U, call);
}

template <> struct OpenCLOperator<backend::Bias> final {
  static bool Make(const backend::Bias &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(!ConfigureBias("Bias", args.input, args.bias, args.element_count,
                                      args.row_size, args.output, call),
                       false);
    BindOpenCLArguments(call, args.input, args.bias, args.element_count, args.row_size,
                        args.output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::BiasResidual> final {
  static bool Make(const backend::BiasResidual &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(!ConfigureBias("BiasResidual", args.input, args.bias, args.element_count,
                                      args.row_size, args.output, call),
                       false);
    OMNI_RETURN_VAL_IF_LOG(
        !HasElements(args.residual, args.element_count, ElementBytes(call->type)), false, ERROR,
        "Invalid bias residual buffer");
    BindOpenCLArguments(call, args.input, args.bias, args.residual, args.element_count,
                        args.row_size, args.output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::GeluBias> final {
  static bool Make(const backend::GeluBias &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF_LOG(args.is_ggml_gelu && !args.is_approximate, false, ERROR,
                           "GGML GELU requires the approximate activation");
    OMNI_RETURN_VAL_IF(!ConfigureBias("GeluBias", args.input, args.bias, args.element_count,
                                      args.row_size, args.output, call),
                       false);
    BindOpenCLArguments(call, args.input, args.bias, args.element_count, args.row_size,
                        args.is_approximate, args.is_ggml_gelu, args.output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::GridQkvRope> final {
  static bool Make(const backend::GridQkvRope &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.grid_height <= 0 || args.grid_width <= 0 ||
                               args.merge_size <= 0 || args.head_count <= 0 ||
                               args.head_size <= 0 || args.head_size % 4 != 0 ||
                               int64_t(args.grid_height) * args.grid_width != args.token_count ||
                               args.grid_height % args.merge_size != 0 ||
                               args.grid_width % args.merge_size != 0,
                           false, ERROR, "Invalid vision rotary dimensions");
    const uint64_t count = uint64_t(args.token_count) * args.head_count * args.head_size;
    const size_t bytes = ElementBytes(call->type);
    const size_t key_value_bytes = args.is_fp16_key_value ? sizeof(uint16_t) : bytes;
    OMNI_RETURN_VAL_IF_LOG(
        !HasElements(args.qkv, count * 3U, bytes) || !HasElements(args.query, count, bytes) ||
            !HasElements(args.key, count, key_value_bytes) ||
            !HasElements(args.value, count, key_value_bytes) ||
            (args.is_fp16_key_value && call->type != backend::ScalarType::kFloat32),
        false, ERROR, "Invalid vision rotary buffers");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("vision_encoder_kernel/vision_encoder_kernel.cl",
                                        "GridQkvRope", (count + 255U) / 256U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.qkv, args.token_count, args.grid_height, args.grid_width,
                        args.merge_size, args.head_count, args.head_size);
    if (args.is_ggml_rope) {
      call->build_defines.push_back("OMNI_GGML_ROPE");
      BindOpenCLArguments(call, std::pow(10000.0F, -2.0F / static_cast<float>(args.head_size / 2)));
    }
    BindOpenCLArguments(call, args.query, args.key, args.value);
    if (args.is_fp16_key_value) {
      call->build_defines.push_back("OMNI_FP16_KEY_VALUE");
    }
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
