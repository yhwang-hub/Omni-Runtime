#pragma once

#include "kernels/operator_runtime_common.h"

#include "kernels/opencl_operator_runtime.h"
#include "kernels/gqa_attention_kernel/attention_define.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::Attention> final {
  static bool Make(const backend::Attention &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(!IsValidAttentionShape(args.query_token_count, args.query_head_count,
                                                  args.key_value_head_count,
                                                  args.cached_token_count, args.head_dim) ||
                               args.start_position < 0,
                           false, ERROR, "Invalid attention dimensions");
    const size_t bytes = ElementBytes(call->type);
    const size_t key_value_bytes = args.is_fp16_key_value ? sizeof(uint16_t) : bytes;
    const uint64_t query_count =
        uint64_t(args.query_token_count) * args.query_head_count * args.head_dim;
    const uint64_t cache_count =
        uint64_t(args.cached_token_count) * args.key_value_head_count * args.head_dim;
    OMNI_RETURN_VAL_IF_LOG(
        !HasElements(args.query, query_count, bytes) ||
            !HasElements(args.key_cache, cache_count, key_value_bytes) ||
            !HasElements(args.value_cache, cache_count, key_value_bytes) ||
            !HasElements(args.output, query_count, bytes) ||
            (args.is_fp16_key_value && ((!args.is_fp16_accumulation && !args.is_ggml_softmax) ||
                                        call->type != backend::ScalarType::kFloat32)) ||
            (args.cache_position.IsValid() &&
             (!HasElements(args.cache_position, 1U, sizeof(int32_t)) ||
              args.query_token_count != 1 || !args.is_causal)),
        false, ERROR, "Invalid attention operation");
    OMNI_RETURN_VAL_IF_LOG(
        args.is_ggml_softmax &&
            (!args.is_fp16_attention || call->type != backend::ScalarType::kFloat32 ||
             !args.is_causal || args.head_dim != 256 || !call->has_nvidia_mma ||
             args.query_head_count > args.key_value_head_count * 8 ||
             call->compute_unit_count <= 0),
        false, ERROR, "GGML softmax requires FP32 causal tensor attention with head size 256");
    if (args.is_fp16_accumulation) {
      OMNI_RETURN_VAL_IF_LOG(
          args.head_dim != 72 || args.is_causal ||
              args.query_token_count != args.cached_token_count ||
              args.query_head_count != args.key_value_head_count || args.cache_position.IsValid(),
          false, ERROR, "FP16 accumulation requires full self-attention with head size 72");
      const int32_t queries = call->has_nvidia_mma ? 64 : 32;
      OMNI_RETURN_VAL_IF(
          !ConfigureOpenCL("vision_encoder_kernel/vision_half_attention.cl", "VisionHalfAttention",
                           uint64_t((args.query_token_count + queries - 1) / queries) *
                               args.query_head_count,
                           static_cast<size_t>(queries * 4), call),
          false);
      BindOpenCLArguments(call, args.query, args.key_cache, args.value_cache,
                          args.query_token_count, args.query_head_count, args.output);
      if (args.is_fp16_key_value) {
        call->build_defines.push_back("OMNI_FP16_KEY_VALUE");
      }
      return true;
    }
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("gqa_attention_kernel/gqa_attention_kernel.cl", "Attention",
                                        uint64_t(args.query_token_count) * args.query_head_count,
                                        128U, call),
                       false);
    BindOpenCLArguments(call, args.query, args.key_cache, args.value_cache, args.query_token_count,
                        args.query_head_count, args.key_value_head_count, args.cached_token_count,
                        args.start_position, args.head_dim, args.is_causal, args.output,
                        args.decode_partials, args.cache_position, args.decode_partials.IsValid(),
                        args.cache_position.IsValid());
    call->tensor_shape = {args.query_token_count, args.query_head_count, args.head_dim};
    if (args.is_fp16_attention) {
      call->build_defines.push_back("OMNI_HALF_ATTENTION");
    }
    if (args.is_fp16_key_value) {
      call->build_defines.push_back("OMNI_FP16_KEY_VALUE");
    }
    if (args.is_ggml_softmax) {
      call->build_defines.push_back("OMNI_GGML_SOFTMAX");
      call->build_defines.push_back("OMNI_GGML_STREAM_GROUPS=" +
                                    std::to_string(GgmlAttentionStreamGroups(
                                        args.query_token_count, args.cached_token_count,
                                        args.key_value_head_count, call->compute_unit_count)));
    }
    if (call->has_nvidia_mma &&
        (args.is_fp16_attention || call->type != backend::ScalarType::kFloat32) &&
        (args.query_token_count > 1 || args.is_ggml_softmax) &&
        (args.head_dim <= 128 || args.is_fp16_attention)) {
      const size_t query_tile = args.is_ggml_softmax ? 8U : args.head_dim <= 80 ? 128U : 64U;
      const size_t work_items = args.is_ggml_softmax ? 32U : query_tile * 2U;
      call->entry = "AttentionTensor";
      call->dimensions = 2U;
      call->local_size = {work_items, 1U, 1U};
      call->global_size = {(static_cast<size_t>(args.query_token_count) + query_tile - 1U) /
                               query_tile * work_items,
                           static_cast<size_t>(args.query_head_count), 1U};
      call->build_defines.push_back(args.head_dim <= 80    ? "OMNI_ATTENTION_DIM=80"
                                    : args.head_dim <= 128 ? "OMNI_ATTENTION_DIM=128"
                                                           : "OMNI_ATTENTION_DIM=256");
    }
    if (!args.is_ggml_softmax && args.query_token_count == 1 && args.is_causal &&
        (args.cache_position.IsValid() || args.start_position == args.cached_token_count - 1) &&
        args.decode_partials.IsValid()) {
      const uint64_t partitions = (uint64_t(args.cached_token_count) + 127U) / 128U;
      OMNI_RETURN_VAL_IF_LOG(!HasElements(args.decode_partials,
                                          partitions * args.query_head_count * (args.head_dim + 2U),
                                          sizeof(float)),
                             false, ERROR, "Decode attention partials are too small");
      const bool use_grouped =
          call->has_nvidia_mma && call->type != backend::ScalarType::kFloat32 &&
          args.head_dim == 128 && args.query_head_count / args.key_value_head_count == 4;
      call->source = "gqa_attention_kernel/decode_attention.cl";
      call->entry = use_grouped ? "AttentionDecodeGrouped" : "AttentionDecode";
      call->dimensions = 2U;
      call->local_size = {128U, 1U, 1U};
      call->global_size = {
          static_cast<size_t>(use_grouped ? args.key_value_head_count : args.query_head_count) *
              128U,
          static_cast<size_t>(partitions), 1U};
    }
    if (args.is_ggml_softmax && args.query_token_count == 1) {
      OMNI_RETURN_VAL_IF_LOG(call->compute_unit_count % args.key_value_head_count != 0, false,
                             ERROR, "GGML decode requires uniformly partitioned tensor attention");
      const int32_t partition_cap = call->compute_unit_count / args.key_value_head_count;
      const uint64_t partitions = std::min<uint64_t>(
          partition_cap, ((uint64_t(args.cached_token_count) + 255U) / 256U) * 4U);
      OMNI_RETURN_VAL_IF_LOG(!HasElements(args.decode_partials,
                                          partitions * args.query_head_count * 258U, sizeof(float)),
                             false, ERROR, "GGML decode partials are too small");
      call->source = "gqa_attention_kernel/decode_attention.cl";
      call->entry = "AttentionGgmlDecode";
      call->dimensions = 2U;
      call->local_size = {128U, 1U, 1U};
      call->global_size = {static_cast<size_t>(args.key_value_head_count) * 128U,
                           static_cast<size_t>(partitions), 1U};
      call->build_defines.push_back("OMNI_GGML_PARTITION_CAP=" + std::to_string(partition_cap));
    }
    return true;
  }

  static bool Run(const backend::OpenCLInvocation &invocation,
                  backend::OpenCLExecutor *const executor) {
    OMNI_RETURN_VAL_IF(executor == nullptr || !executor->Launch(invocation), false);
    if (invocation.entry == "AttentionDecode" || invocation.entry == "AttentionDecodeGrouped" ||
        invocation.entry == "AttentionGgmlDecode") {
      backend::OpenCLInvocation merge = invocation;
      merge.entry = invocation.entry == "AttentionGgmlDecode" ? "AttentionGgmlDecodeMerge"
                                                              : "AttentionDecodeMerge";
      merge.dimensions = 1U;
      merge.local_size = {256U, 1U, 1U};
      merge.global_size = {static_cast<size_t>(merge.tensor_shape[1]) * 256U, 1U, 1U};
      return executor->Launch(merge);
    }
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
