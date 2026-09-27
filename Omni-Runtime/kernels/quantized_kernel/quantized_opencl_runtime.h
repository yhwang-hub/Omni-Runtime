#pragma once

#include "kernels/opencl_operator_runtime.h"
#include "kernels/quantized_kernel/quantized_runtime_utils.h"

namespace omni_runtime::kernels::runtime {
template <> struct OpenCLOperator<backend::QuantizedLinear> final {
  static bool Make(const backend::QuantizedLinear &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(!ValidateQuantized(args, ElementBytes(call->type)), false, ERROR,
                           "Invalid GGUF linear buffers or storage type");
    const bool is_scaled = args.activation_scales.IsValid();
    OMNI_RETURN_VAL_IF_LOG(args.is_packed_q8 && !call->has_nvidia_mma, false, ERROR,
                           "Packed Q8 input requires NVIDIA integer tensor support");
    const bool is_gemv =
        is_scaled && args.rows <= 8 && args.activation_block_size == 32 && call->has_nvidia_mma;
    const bool is_tensor = args.rows > 1 && call->has_nvidia_mma && !is_gemv;
    const bool is_bfloat =
        is_tensor && is_scaled && args.weight_type == 40 && args.activation_block_size == 16;
    const bool is_q8 =
        is_tensor && is_scaled && args.weight_type == 8 && args.activation_block_size == 32;
    const bool is_packed_tensor = is_bfloat && args.is_packed_nvfp4;
    const bool has_small_fp4_tile = is_packed_tensor && args.rows <= 16;
    OMNI_RETURN_VAL_IF_LOG((is_q8 || is_packed_tensor) && args.is_ggml_split_k &&
                               call->compute_unit_count <= 0,
                           false, ERROR, "GGML split-K requires a valid device compute unit count");
    const bool is_split_k =
        (is_q8 || is_packed_tensor) && NeedGgmlSplitK(args, call->compute_unit_count);
    call->source = "quantized_kernel/quantized_kernel.cl";
    if (is_gemv) {
      call->build_defines.push_back("OMNI_GEMV_TYPE=" + std::to_string(args.weight_type));
    }
    if (is_tensor && !is_scaled) {
      call->build_defines.push_back("OMNI_DENSE_TYPE=" + std::to_string(args.weight_type));
    }
    call->entry = is_scaled ? (is_gemv     ? "QuantizedScaledGemv"
                               : is_bfloat ? "QuantizedBfloatTensor"
                               : is_q8     ? "QuantizedQ8Tensor"
                               : is_tensor ? "QuantizedScaledTensor"
                                           : "QuantizedScaledLinear")
                            : (is_tensor ? "QuantizedTensor" : "QuantizedLinear");
    call->dimensions = 2U;
    if (args.is_packed_nvfp4) {
      call->build_defines.push_back("OMNI_PACKED_NVFP4");
    }
    if (has_small_fp4_tile) {
      call->build_defines.push_back("OMNI_SMALL_FP4_TILE");
    }
    if (args.is_packed_q8) {
      call->build_defines.push_back("OMNI_PACKED_Q8");
    }
    if (is_split_k) {
      call->build_defines.push_back("OMNI_GGML_SPLIT_K");
      if (is_q8 && GgmlTokenTile(args.rows) % 64 == 0) {
        call->build_defines.push_back("OMNI_GGML_LARGE_SPLIT_TILE");
      }
    }
    const size_t group_size = has_small_fp4_tile ? 32U : is_gemv || is_packed_tensor ? 256U : 128U;
    const int32_t gemv_warps = GgmlGemvWarps(args);
    const int32_t gemv_columns = 8 / gemv_warps;
    const int32_t tile_columns = is_packed_tensor ? 32 : is_bfloat ? 128 : 64;
    const int32_t groups = is_gemv     ? (args.columns + gemv_columns - 1) / gemv_columns
                           : is_tensor ? (args.columns + tile_columns - 1) / tile_columns
                                       : args.columns;
    call->local_size = {group_size, 1U, 1U};
    const int32_t tile_rows = is_packed_tensor && is_split_k ? GgmlTokenTile(args.rows)
                              : is_split_k && GgmlTokenTile(args.rows) % 64 != 0 ? 8
                              : has_small_fp4_tile                               ? 16
                              : args.is_packed_nvfp4                             ? 128
                                                                                 : 64;
    const int32_t row_groups = is_tensor ? (args.rows + tile_rows - 1) / tile_rows : args.rows;
    const bool is_token_first = is_q8 || is_bfloat;
    call->global_size = {static_cast<size_t>(is_token_first ? row_groups : groups) * group_size,
                         static_cast<size_t>(is_token_first ? groups : row_groups), 1U};
    BindOpenCLArguments(call, args.input, args.weight, args.weight_type, args.weight_scale,
                        args.rows, args.inner, args.columns, args.output);
    if (is_scaled) {
      BindOpenCLArguments(call, args.activation_scales, args.activation_block_size);
    }
    if (is_q8 || is_packed_tensor) {
      BindOpenCLArguments(call, is_split_k ? GgmlTokenTile(args.rows) : 0,
                          is_split_k ? GgmlStreamGroups(args, call->compute_unit_count) : 0);
    }
    if (is_gemv) {
      call->build_defines.push_back("OMNI_GEMV_WARPS=" + std::to_string(gemv_warps));
      call->Local(!args.is_packed_q8 && args.inner <= 32768 ? static_cast<size_t>(args.inner)
                                                            : sizeof(uint32_t));
    }
    return true;
  }
};

template <> struct OpenCLOperator<backend::QuantizeActivation> final {
  static bool Make(const backend::QuantizeActivation &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(!ValidateQuantized(args, ElementBytes(call->type)), false, ERROR,
                           "Invalid activation quantization buffers or dimensions");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("quantized_kernel/quantized_kernel.cl",
                                        "QuantizeActivation", args.rows, 256U, call),
                       false);
    call->dimensions = 2U;
    call->global_size[1] = QuantizeColumnGroups(args);
    BindOpenCLArguments(call, args.input, args.weight_type, args.rows, args.columns, args.is_decode,
                        args.is_ggml_rounding, args.is_packed_nvfp4, args.is_packed_q8, args.output,
                        args.scales);
    return true;
  }
};

template <> struct OpenCLOperator<backend::QuantizedEmbedding> final {
  static bool Make(const backend::QuantizedEmbedding &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(!ValidateQuantized(args, ElementBytes(call->type)), false, ERROR,
                           "Invalid GGUF embedding buffers or storage type");
    const uint64_t count = static_cast<uint64_t>(args.token_count) * args.hidden_size;
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("quantized_kernel/quantized_kernel.cl",
                                        "QuantizedEmbedding", (count + 255U) / 256U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.indices, args.weight, args.weight_type, args.token_count,
                        args.hidden_size, args.vocabulary_size, args.output);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
