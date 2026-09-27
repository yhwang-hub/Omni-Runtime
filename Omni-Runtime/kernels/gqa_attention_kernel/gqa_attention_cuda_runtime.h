#pragma once

#include "kernels/operator_runtime_common.h"

#include "kernels/cuda_operator_runtime.h"
#include "kernels/flash_attention_v4_kernel/flash_attention_v4_cuda_runtime.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <type_traits>

#include "kernels/kernel_runtime.h"
#include "kernels/gqa_attention_kernel/attention_define.h"
#include "kernels/flash_attention_v2_kernel/flash_attention_v2_cuda_runtime.h"
#include "kernels/flash_attention_kernel/flash_attention_cuda_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class GqaAttentionKernelRuntime final {
public:
  using Args = AttentionArgs;

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(!IsValidAttention(args), false);
    if (args.query_token_count == 1 && args.decode_partials != nullptr) {
      const int32_t cached_token_count =
          args.is_causal && args.cache_position == nullptr
              ? static_cast<int32_t>(std::min<int64_t>(
                    args.cached_token_count, static_cast<int64_t>(args.start_position) + 1))
              : args.cached_token_count;
      const int32_t partitions = (cached_token_count + 127) / 128;
      const bool is_grouped =
          sizeof(T) == 2U && !std::is_integral_v<T> && CudaKernelRuntime::HasTensorMma() &&
          args.query_head_count / args.key_value_head_count == 4 && args.head_dim == 128 &&
          reinterpret_cast<uintptr_t>(args.query) % 4U == 0U &&
          reinterpret_cast<uintptr_t>(args.key_cache) % 16U == 0U &&
          reinterpret_cast<uintptr_t>(args.value_cache) % 16U == 0U;
      return CudaKernelRuntime::CompileAndLaunch(
                 tag + "_split",
                 CudaKernelRuntime::MakeKernelSource<T>(
                     "kernels/gqa_attention_kernel/decode_attention.cuh",
                     is_grouped
                         ? "omni_runtime::kernels::gqa_attention_kernel::GroupedDecodeAttentionImpl"
                         : "omni_runtime::kernels::gqa_attention_kernel::DecodeAttentionImpl"),
                 {.grid_dim = {static_cast<uint32_t>(is_grouped ? args.key_value_head_count
                                                                : args.query_head_count),
                               static_cast<uint32_t>(partitions), 1U},
                  .block_dim = {128U, 1U, 1U},
                  .shared_memory_bytes = 0},
                 static_cast<const T *>(args.query), static_cast<const T *>(args.key_cache),
                 static_cast<const T *>(args.value_cache), args.query_head_count,
                 args.key_value_head_count, cached_token_count, args.head_dim,
                 static_cast<float *>(args.decode_partials), args.cache_position) &&
             CudaKernelRuntime::CompileAndLaunch(
                 tag + "_merge",
                 CudaKernelRuntime::MakeKernelSource<T>(
                     "kernels/gqa_attention_kernel/decode_attention.cuh",
                     "omni_runtime::kernels::gqa_attention_kernel::MergeDecodeAttentionImpl"),
                 {.grid_dim = {static_cast<uint32_t>(args.query_head_count), 1U, 1U},
                  .block_dim = {256U, 1U, 1U},
                  .shared_memory_bytes = 0},
                 static_cast<const float *>(args.decode_partials), args.query_head_count,
                 partitions, args.head_dim, static_cast<T *>(args.output), args.cache_position);
    }
    if constexpr (sizeof(T) == 2U && !std::is_integral_v<T>) {
      if (args.query_token_count > 1 && args.head_dim <= 128 && CudaKernelRuntime::HasTensorMma()) {
        return FlashAttentionV2KernelRuntime<T>::CompileAndLaunch(tag + "_tiled", args);
      }
    }
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/gqa_attention_kernel/gqa_attention_kernel.cuh",
            "omni_runtime::kernels::gqa_attention_kernel::GqaAttentionImpl"),
        {.grid_dim = {(static_cast<uint32_t>(args.query_token_count) + 3U) / 4U,
                      static_cast<uint32_t>(args.query_head_count), 1U},
         .block_dim = {128U, 1U, 1U},
         .shared_memory_bytes = FlashAttentionKernelRuntime<T>::SharedMemoryBytes(args.head_dim)},
        static_cast<const T *>(args.query), static_cast<const T *>(args.key_cache),
        static_cast<const T *>(args.value_cache), args.query_token_count, args.query_head_count,
        args.key_value_head_count, args.cached_token_count, args.start_position, args.head_dim,
        args.is_causal, static_cast<T *>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::Attention> final {
  static bool LaunchGgmlDecode(const AttentionArgs &args) {
    OMNI_RETURN_VAL_IF(!IsValidAttention(args), false);
    const int32_t compute_units = CudaKernelRuntime::MultiprocessorCount();
    OMNI_RETURN_VAL_IF_LOG(
        !CudaKernelRuntime::HasTensorMma() || compute_units <= 0 ||
            compute_units % args.key_value_head_count != 0 || args.decode_partials == nullptr,
        false, ERROR, "GGML decode requires uniformly partitioned tensor attention");
    const int32_t partition_cap = compute_units / args.key_value_head_count;
    const int32_t partitions = static_cast<int32_t>(std::min<int64_t>(
        partition_cap, ((static_cast<int64_t>(args.cached_token_count) + 255) / 256) * 4));
    const std::string source = "kernels/gqa_attention_kernel/decode_attention.cuh";
    const std::string kernel =
        "omni_runtime::kernels::gqa_attention_kernel::GgmlDecodeAttentionImpl";
    return CudaKernelRuntime::CompileAndLaunch(
               "ggml_decode_attention",
               args.is_fp16_key_value ? CudaKernelRuntime::MakeKernelSource<T, 1>(source, kernel)
                                      : CudaKernelRuntime::MakeKernelSource<T, 0>(source, kernel),
               {.grid_dim = {static_cast<uint32_t>(args.key_value_head_count),
                             static_cast<uint32_t>(partitions), 1U},
                .block_dim = {128U, 1U, 1U},
                .shared_memory_bytes = 0},
               static_cast<const T *>(args.query), static_cast<const T *>(args.key_cache),
               static_cast<const T *>(args.value_cache), args.query_head_count,
               args.key_value_head_count, args.cached_token_count, args.start_position,
               partition_cap, args.cache_position, static_cast<float *>(args.decode_partials)) &&
           CudaKernelRuntime::CompileAndLaunch(
               "ggml_decode_attention_merge",
               CudaKernelRuntime::MakeKernelSource<T>(
                   source,
                   "omni_runtime::kernels::gqa_attention_kernel::MergeGgmlDecodeAttentionImpl"),
               {.grid_dim = {static_cast<uint32_t>(args.query_head_count), 1U, 1U},
                .block_dim = {256U, 1U, 1U},
                .shared_memory_bytes = 0},
               static_cast<const float *>(args.decode_partials), args.query_head_count,
               args.cached_token_count, partition_cap, args.cache_position,
               static_cast<T *>(args.output));
  }

  template <int32_t kDimensions, int32_t kKeys, int32_t kIsGgml>
  static bool LaunchHalf(const AttentionArgs &args) {
    OMNI_RETURN_VAL_IF_LOG(!IsValidAttention(args) || !CudaKernelRuntime::HasTensorMma(), false,
                           ERROR, "Mixed precision attention requires CUDA tensor MMA");
    constexpr uint32_t kQueries = kIsGgml != 0 ? 8U : kDimensions <= 80 ? 128U : 64U;
    const int32_t compute_units = kIsGgml != 0 ? CudaKernelRuntime::MultiprocessorCount() : 0;
    OMNI_RETURN_VAL_IF_LOG(kIsGgml != 0 && compute_units <= 0, false, ERROR,
                           "GGML attention requires a valid device compute unit count");
    const int32_t stream_groups =
        kIsGgml != 0 ? GgmlAttentionStreamGroups(args.query_token_count, args.cached_token_count,
                                                 args.key_value_head_count, compute_units)
                     : 0;
    const std::string include = "kernels/flash_attention_v2_kernel/flash_attention_v2_kernel.cuh";
    const std::string kernel =
        "omni_runtime::kernels::flash_attention_v2_kernel::HalfAttentionImpl";
    return CudaKernelRuntime::CompileAndLaunch(
        "half_attention",
        args.is_fp16_key_value
            ? CudaKernelRuntime::MakeKernelSource<T, kDimensions, kKeys, kIsGgml, 1>(include,
                                                                                     kernel)
            : CudaKernelRuntime::MakeKernelSource<T, kDimensions, kKeys, kIsGgml, 0>(include,
                                                                                     kernel),
        {.grid_dim = {(static_cast<uint32_t>(args.query_token_count) + kQueries - 1U) / kQueries,
                      static_cast<uint32_t>(args.query_head_count), 1U},
         .block_dim = {kIsGgml != 0 ? 32U : kQueries * 2U, 1U, 1U},
         .shared_memory_bytes = 0},
        static_cast<const T *>(args.query), static_cast<const T *>(args.key_cache),
        static_cast<const T *>(args.value_cache), args.query_token_count, args.query_head_count,
        args.key_value_head_count, args.cached_token_count, args.start_position, args.head_dim,
        args.is_causal, stream_groups, args.cache_position, static_cast<T *>(args.output));
  }

  static bool Launch(const backend::Attention &args) {
    OMNI_RETURN_VAL_IF_LOG(
        !args.query.IsValid() || !args.key_cache.IsValid() || !args.value_cache.IsValid() ||
            !args.output.IsValid() ||
            ((!args.decode_partials.IsValid() ||
              args.decode_partials.backend_kind != backend::BackendKind::kCuda) &&
             (args.decode_partials.data != nullptr || args.decode_partials.offset_bytes != 0U ||
              args.decode_partials.size_bytes != 0U)) ||
            ((!args.cache_position.IsValid() ||
              args.cache_position.backend_kind != backend::BackendKind::kCuda) &&
             (args.cache_position.data != nullptr || args.cache_position.offset_bytes != 0U ||
              args.cache_position.size_bytes != 0U)),
        false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(!IsValidAttentionShape(args.query_token_count, args.query_head_count,
                                                  args.key_value_head_count,
                                                  args.cached_token_count, args.head_dim),
                           false, ERROR, "Invalid compute operation dimensions");
    OMNI_RETURN_VAL_IF_LOG(
        args.is_fp16_key_value &&
            ((!args.is_fp16_accumulation && !args.is_ggml_softmax) || !std::is_same_v<T, float>),
        false, ERROR, "FP16 key/value storage requires FP32 mixed attention");
    const uint64_t query_count =
        uint64_t(args.query_token_count) * args.query_head_count * args.head_dim;
    const uint64_t cache_count =
        uint64_t(args.cached_token_count) * args.key_value_head_count * args.head_dim;
    const size_t key_value_bytes = args.is_fp16_key_value ? sizeof(uint16_t) : sizeof(T);
    OMNI_RETURN_VAL_IF_LOG(!HasElements(args.query, query_count, sizeof(T)) ||
                               !HasElements(args.key_cache, cache_count, key_value_bytes) ||
                               !HasElements(args.value_cache, cache_count, key_value_bytes) ||
                               !HasElements(args.output, query_count, sizeof(T)),
                           false, ERROR, "Invalid attention tensor sizes");
    const AttentionArgs parameters = {
        .options = {.grid_dim = {static_cast<uint32_t>(uint64_t(args.query_token_count) *
                                                       args.query_head_count),
                                 1U, 1U},
                    .block_dim = {128U, 1U, 1U},
                    .shared_memory_bytes = 0},
        .query = CudaAddress<void>(args.query),
        .key_cache = CudaAddress<void>(args.key_cache),
        .value_cache = CudaAddress<void>(args.value_cache),
        .query_token_count = args.query_token_count,
        .query_head_count = args.query_head_count,
        .key_value_head_count = args.key_value_head_count,
        .cached_token_count = args.cached_token_count,
        .start_position = args.start_position,
        .head_dim = args.head_dim,
        .is_causal = args.is_causal,
        .output = CudaAddress<void>(args.output),
        .decode_partials = CudaAddress<void>(args.decode_partials),
        .cache_position = CudaAddress<const int32_t>(args.cache_position),
        .is_fp16_key_value = args.is_fp16_key_value};
    if (args.is_fp16_accumulation) {
      OMNI_RETURN_VAL_IF_LOG(
          !IsValidAttention(parameters) || args.head_dim != 72 || args.is_causal ||
              args.query_token_count != args.cached_token_count ||
              args.query_head_count != args.key_value_head_count || args.cache_position.IsValid(),
          false, ERROR, "FP16 accumulation requires full self-attention with head size 72");
      const std::string include = "kernels/vision_encoder_kernel/vision_half_attention.cuh";
      const std::string kernel =
          "omni_runtime::kernels::vision_encoder_kernel::VisionHalfAttention";
      return CudaKernelRuntime::CompileAndLaunch(
          "vision_half_acc_attention",
          args.is_fp16_key_value
              ? CudaKernelRuntime::MakeKernelSource<T, 1, 64, 8>(include, kernel)
              : CudaKernelRuntime::MakeKernelSource<T, 0, 64, 8>(include, kernel),
          {.grid_dim = {static_cast<uint32_t>((args.query_token_count + 63) / 64) *
                            static_cast<uint32_t>(args.query_head_count),
                        1U, 1U},
           .block_dim = {256U, 1U, 1U},
           .shared_memory_bytes = 0},
          static_cast<const T *>(parameters.query), parameters.key_cache, parameters.value_cache,
          args.query_token_count, args.query_head_count, static_cast<T *>(parameters.output));
    }
    if (args.is_ggml_softmax) {
      OMNI_RETURN_VAL_IF_LOG(
          (!args.is_fp16_attention || !std::is_same_v<T, float> || !args.is_causal ||
           args.head_dim != 256 || args.query_head_count > args.key_value_head_count * 8),
          false, ERROR, "GGML softmax requires FP32 causal attention with head size 256");
      if (args.query_token_count == 1) {
        const int32_t compute_units = CudaKernelRuntime::MultiprocessorCount();
        const uint64_t partitions =
            std::min<uint64_t>(compute_units > 0 ? compute_units / args.key_value_head_count : 0,
                               ((uint64_t(args.cached_token_count) + 255U) / 256U) * 4U);
        OMNI_RETURN_VAL_IF_LOG(!HasElements(args.decode_partials,
                                            partitions * args.query_head_count * 258U,
                                            sizeof(float)),
                               false, ERROR, "GGML decode partials are too small");
        return LaunchGgmlDecode(parameters);
      }
      return LaunchHalf<256, 32, 1>(parameters);
    }
    if (args.is_fp16_attention && args.query_token_count > 1) {
      if (args.head_dim <= 80) {
        return LaunchHalf<80, 64, 0>(parameters);
      }
      return args.head_dim <= 128 ? LaunchHalf<128, 64, 0>(parameters)
                                  : LaunchHalf<256, 32, 0>(parameters);
    }
    if constexpr (sizeof(T) == 2U) {
      if (args.query_token_count > 1 && args.head_dim <= 128 &&
          kernel::jit::JitRuntime::HasSm120()) {
        return FlashAttentionV4KernelRuntime<T>::CompileAndLaunch("backend_Attention_v4_sm120",
                                                                  parameters);
      }
    }
    return GqaAttentionKernelRuntime<T>::CompileAndLaunch("backend_Attention", parameters);
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
