#pragma once

#include "kernels/cuda_operator_runtime.h"
#include "kernels/quantized_kernel/quantized_runtime_utils.h"

namespace omni_runtime::kernels::runtime {
template <typename T> struct CudaOperator<T, backend::QuantizedLinear> final {
  static bool Launch(const backend::QuantizedLinear &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateQuantized(args, sizeof(T)), false, ERROR,
                           "Invalid GGUF linear buffers or storage type");
    const T *const input = CudaAddress<const T>(args.input);
    const uint8_t *const weight = CudaAddress<const uint8_t>(args.weight);
    T *const output = CudaAddress<T>(args.output);
    OMNI_RETURN_VAL_IF(input == nullptr || weight == nullptr || output == nullptr, false);
    if (args.activation_scales.IsValid()) {
      return args.weight_type == 8 ? LaunchScaled<8>(args) : LaunchScaled<40>(args);
    }
    if (args.rows == 1) {
      return CudaKernelRuntime::CompileAndLaunch(
          "gguf_gemv",
          CudaKernelRuntime::MakeKernelSource<T>(
              "kernels/quantized_kernel/quantized_kernel.cuh",
              "omni_runtime::kernels::quantized_kernel::GemvImpl"),
          {.grid_dim = {static_cast<uint32_t>((args.columns + 7) / 8), 1U, 1U},
           .block_dim = {256U, 1U, 1U},
           .shared_memory_bytes = 0},
          input, weight, args.weight_type, args.weight_scale, args.inner, args.columns, output);
    }
    OMNI_RETURN_VAL_IF_LOG(!CudaKernelRuntime::HasTensorMma(), false, ERROR,
                           "GGUF prefill requires CUDA tensor MMA support");
    return CudaKernelRuntime::CompileAndLaunch(
        "gguf_matmul",
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/quantized_kernel/quantized_kernel.cuh",
            "omni_runtime::kernels::quantized_kernel::MatmulImpl"),
        {.grid_dim = {static_cast<uint32_t>((args.columns + 63) / 64),
                      static_cast<uint32_t>((args.rows + 63) / 64), 1U},
         .block_dim = {128U, 1U, 1U},
         .shared_memory_bytes = 0},
        input, weight, args.weight_type, args.weight_scale, args.rows, args.inner, args.columns,
        output);
  }

private:
  template <int32_t kType> static bool LaunchScaled(const backend::QuantizedLinear &args) {
    const float *const scales = CudaAddress<const float>(args.activation_scales);
    OMNI_RETURN_VAL_IF(scales == nullptr, false);
    if (args.is_packed_nvfp4) {
      const bool has_fp4_mma = CudaKernelRuntime::HasSm120();
      if (has_fp4_mma) {
        const int32_t compute_units =
            args.is_ggml_split_k ? CudaKernelRuntime::MultiprocessorCount() : 0;
        OMNI_RETURN_VAL_IF_LOG(args.is_ggml_split_k && compute_units <= 0, false, ERROR,
                               "GGML split-K requires a valid device compute unit count");
        const bool is_split_k = NeedGgmlSplitK(args, compute_units);
        const bool has_small_tile = args.rows <= 16;
        const bool has_two_part_fixup = is_split_k && !has_small_tile && std::is_same_v<T, float> &&
                                        HasAtMostTwoGgmlParts(args, compute_units);
        const int32_t physical_rows = has_small_tile ? 16 : 128;
        const int32_t token_tile = is_split_k ? GgmlTokenTile(args.rows) : physical_rows;
        const int32_t tile_columns = has_small_tile ? 32 : is_split_k ? 64 : 128;
        return CudaKernelRuntime::CompileAndLaunch(
            "gguf_packed_nvfp4",
            is_split_k ? (has_small_tile       ? MakeNvfp4Source<1, 32, 16>(args)
                          : has_two_part_fixup ? MakeNvfp4Source<2, 64, 128>(args)
                                               : MakeNvfp4Source<1, 64, 128>(args))
                       : (has_small_tile ? MakeNvfp4Source<0, 32, 16>(args)
                                         : MakeNvfp4Source<0, 128, 128>(args)),
            {.grid_dim = {static_cast<uint32_t>((args.rows + token_tile - 1) / token_tile),
                          static_cast<uint32_t>((args.columns + tile_columns - 1) / tile_columns),
                          1U},
             .block_dim = {static_cast<uint32_t>(physical_rows * 2), 1U, 1U},
             .shared_memory_bytes = 0},
            CudaAddress<const uint8_t>(args.input), CudaAddress<const uint8_t>(args.weight), scales,
            args.weight_scale, args.rows, args.inner, args.columns, token_tile,
            is_split_k ? GgmlStreamGroups(args, compute_units) : 0, CudaAddress<T>(args.output));
      }
      return CudaKernelRuntime::CompileAndLaunch(
          "gguf_packed_nvfp4",
          CudaKernelRuntime::MakeKernelSource<T>(
              "kernels/quantized_kernel/quantized_kernel.cuh",
              "omni_runtime::kernels::quantized_kernel::PackedNvfp4GemvImpl"),
          {.grid_dim = {static_cast<uint32_t>((args.columns + 3) / 4),
                        static_cast<uint32_t>(args.rows), 1U},
           .block_dim = {128U, 1U, 1U},
           .shared_memory_bytes = 0},
          CudaAddress<const uint8_t>(args.input), CudaAddress<const uint8_t>(args.weight), scales,
          args.weight_scale, args.rows, args.inner, args.columns, CudaAddress<T>(args.output));
    }
    if (args.rows <= 8 && args.activation_block_size == 32) {
      if (GgmlGemvWarps(args) == 8) {
        return LaunchGemv<kType, 8>(args, scales);
      }
      return args.rows <= 4 ? LaunchGemv<kType, 4>(args, scales)
                            : LaunchGemv<kType, 2>(args, scales);
    }
    if constexpr (kType == 8) {
      if (args.activation_block_size == 32) {
        const int32_t compute_units =
            args.is_ggml_split_k ? CudaKernelRuntime::MultiprocessorCount() : 0;
        const bool is_split_k = NeedGgmlSplitK(args, compute_units);
        const int32_t token_tile = is_split_k ? GgmlTokenTile(args.rows) : 0;
        const bool has_large_split_tile = is_split_k && token_tile % 64 == 0;
        const int32_t tile_rows = !is_split_k ? 128 : has_large_split_tile ? 64 : 8;
        const int32_t tile_columns = has_large_split_tile ? 64 : 128;
        return CudaKernelRuntime::CompileAndLaunch(
            "gguf_q8_matmul",
            args.is_packed_q8 ? (has_large_split_tile ? MakeQ8Source<2, 1, 64>(args)
                                 : is_split_k         ? MakeQ8Source<1, 1, 128>(args)
                                                      : MakeQ8Source<0, 1, 128>(args))
                              : (has_large_split_tile ? MakeQ8Source<2, 0, 64>(args)
                                 : is_split_k         ? MakeQ8Source<1, 0, 128>(args)
                                                      : MakeQ8Source<0, 0, 128>(args)),
            {.grid_dim = {static_cast<uint32_t>((args.rows + tile_rows - 1) / tile_rows),
                          static_cast<uint32_t>((args.columns + tile_columns - 1) / tile_columns),
                          1U},
             .block_dim = {has_large_split_tile ? 128U : 256U, 1U, 1U},
             .shared_memory_bytes = 0},
            CudaAddress<const T>(args.input), CudaAddress<const uint8_t>(args.weight), scales,
            args.weight_scale, args.rows, args.inner, args.columns, token_tile,
            is_split_k ? GgmlStreamGroups(args, compute_units) : 0, CudaAddress<T>(args.output));
      }
    }
    return CudaKernelRuntime::CompileAndLaunch(
        "gguf_scaled_matmul",
        CudaKernelRuntime::MakeKernelSource<T, kType>(
            "kernels/quantized_kernel/quantized_kernel.cuh",
            "omni_runtime::kernels::quantized_kernel::ScaledMatmulImpl"),
        {.grid_dim = {static_cast<uint32_t>((args.columns + 63) / 64),
                      static_cast<uint32_t>((args.rows + 63) / 64), 1U},
         .block_dim = {128U, 1U, 1U},
         .shared_memory_bytes = 0},
        CudaAddress<const T>(args.input), CudaAddress<const uint8_t>(args.weight), scales,
        args.activation_block_size, args.weight_scale, args.rows, args.inner, args.columns,
        CudaAddress<T>(args.output));
  }

  template <int32_t kMode, int32_t kIsPacked, int32_t kColumns>
  static std::string MakeQ8Source(const backend::QuantizedLinear &args) {
    const bool is_async = kIsPacked != 0 && args.inner % 64 == 0 &&
                          ((reinterpret_cast<uintptr_t>(CudaAddress<const uint8_t>(args.input)) |
                            reinterpret_cast<uintptr_t>(CudaAddress<const uint8_t>(args.weight))) %
                               4U ==
                           0U) &&
                          CudaKernelRuntime::HasSm120();
    const std::string include = "kernels/quantized_kernel/quantized_kernel.cuh";
    const std::string kernel = "omni_runtime::kernels::quantized_kernel::Q8MatmulImpl";
    return is_async ? CudaKernelRuntime::MakeKernelSource<T, kMode, kIsPacked, kColumns, 1>(include,
                                                                                            kernel)
                    : CudaKernelRuntime::MakeKernelSource<T, kMode, kIsPacked, kColumns, 0>(include,
                                                                                            kernel);
  }

  template <int32_t kMode, int32_t kColumns, int32_t kRows>
  static std::string MakeNvfp4Source(const backend::QuantizedLinear &args) {
    const bool is_async =
        kRows != 16 && ((reinterpret_cast<uintptr_t>(CudaAddress<const uint8_t>(args.input)) |
                         reinterpret_cast<uintptr_t>(CudaAddress<const uint8_t>(args.weight))) %
                            4U ==
                        0U);
    const std::string include = "kernels/quantized_kernel/quantized_kernel.cuh";
    const std::string kernel = "omni_runtime::kernels::quantized_kernel::PackedNvfp4MatmulImpl";
    return is_async
               ? CudaKernelRuntime::MakeKernelSource<T, kMode, kColumns, kRows, 1>(include, kernel)
               : CudaKernelRuntime::MakeKernelSource<T, kMode, kColumns, kRows, 0>(include, kernel);
  }

  template <int32_t kType, int32_t kWarps>
  static bool LaunchGemv(const backend::QuantizedLinear &args, const float *const scales) {
    OMNI_RETURN_VAL_IF(scales == nullptr || args.activation_block_size != 32, false);
    constexpr int32_t kColumns = 8 / kWarps;
    const size_t staged_bytes = static_cast<size_t>(args.inner) / 32U * 34U * kColumns;
    const bool need_stage =
        kType == 8 && args.is_packed_q8 && args.rows == 1 && args.inner % 256 == 0 &&
        staged_bytes <= 16384U &&
        reinterpret_cast<uintptr_t>(CudaAddress<const uint8_t>(args.weight)) % 16U == 0U &&
        CudaKernelRuntime::HasSm120();
    const std::string include = "kernels/quantized_kernel/quantized_kernel.cuh";
    const std::string kernel = "omni_runtime::kernels::quantized_kernel::ScaledGemvImpl";
    return CudaKernelRuntime::CompileAndLaunch(
        "gguf_scaled_gemv",
        need_stage ? CudaKernelRuntime::MakeKernelSource<T, kType, kWarps, 1, 1>(include, kernel)
        : args.is_packed_q8
            ? CudaKernelRuntime::MakeKernelSource<T, kType, kWarps, 1, 0>(include, kernel)
            : CudaKernelRuntime::MakeKernelSource<T, kType, kWarps, 0, 0>(include, kernel),
        {.grid_dim = {static_cast<uint32_t>((args.columns + kColumns - 1) / kColumns),
                      static_cast<uint32_t>(args.rows), 1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = need_stage ? static_cast<uint32_t>(staged_bytes)
                                : !args.is_packed_q8 && args.inner <= 32768
                                    ? static_cast<uint32_t>(args.inner)
                                    : 0U},
        CudaAddress<const T>(args.input), CudaAddress<const uint8_t>(args.weight), scales,
        args.weight_scale, args.inner, args.columns, CudaAddress<T>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::QuantizeActivation> final {
  static bool Launch(const backend::QuantizeActivation &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateQuantized(args, sizeof(T)), false, ERROR,
                           "Invalid activation quantization buffers or dimensions");
    OMNI_RETURN_VAL_IF(CudaAddress<const T>(args.input) == nullptr ||
                           CudaAddress<T>(args.output) == nullptr ||
                           CudaAddress<float>(args.scales) == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        "gguf_quantize_activation",
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/quantized_kernel/quantized_kernel.cuh",
            "omni_runtime::kernels::quantized_kernel::QuantizeImpl"),
        {.grid_dim = {static_cast<uint32_t>(args.rows), QuantizeColumnGroups(args), 1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = 0},
        CudaAddress<const T>(args.input), args.weight_type, args.rows, args.columns, args.is_decode,
        args.is_ggml_rounding, args.is_packed_nvfp4, args.is_packed_q8, CudaAddress<T>(args.output),
        CudaAddress<float>(args.scales));
  }
};

template <typename T> struct CudaOperator<T, backend::QuantizedEmbedding> final {
  static bool Launch(const backend::QuantizedEmbedding &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateQuantized(args, sizeof(T)), false, ERROR,
                           "Invalid GGUF embedding buffers or storage type");
    const int32_t *const indices = CudaAddress<const int32_t>(args.indices);
    const uint8_t *const weight = CudaAddress<const uint8_t>(args.weight);
    T *const output = CudaAddress<T>(args.output);
    OMNI_RETURN_VAL_IF(indices == nullptr || weight == nullptr || output == nullptr, false);
    return CudaKernelRuntime::CompileAndLaunch(
        "gguf_embedding",
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/quantized_kernel/quantized_kernel.cuh",
            "omni_runtime::kernels::quantized_kernel::EmbeddingImpl"),
        {.grid_dim = {static_cast<uint32_t>(std::min<uint64_t>(
                          (static_cast<uint64_t>(args.token_count) * args.hidden_size + 255U) /
                              256U,
                          65535U)),
                      1U, 1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = 0},
        indices, weight, args.weight_type, args.token_count, args.hidden_size, args.vocabulary_size,
        output);
  }
};

} // namespace omni_runtime::kernels::runtime
