#pragma once

#include "kernels/operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::Attention> final {
  static bool Make(const backend::Attention &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const size_t bytes = ElementBytes(call->type);
    const uint64_t query_count =
        uint64_t(args.query_token_count) * args.query_head_count * args.head_dim;
    const uint64_t cache_count =
        uint64_t(args.cached_token_count) * args.key_value_head_count * args.head_dim;
    OMNI_RETURN_VAL_IF_LOG(args.query_token_count <= 0 || args.cached_token_count <= 0 ||
                               args.query_head_count <= 0 || args.key_value_head_count <= 0 ||
                               args.query_head_count % args.key_value_head_count != 0 ||
                               args.head_dim <= 0 || args.head_dim > 256 ||
                               args.start_position < 0 ||
                               !HasElements(args.query, query_count, bytes) ||
                               !HasElements(args.key_cache, cache_count, bytes) ||
                               !HasElements(args.value_cache, cache_count, bytes) ||
                               !HasElements(args.output, query_count, bytes) ||
                               (args.cache_position.IsValid() &&
                                (!HasElements(args.cache_position, 1U, sizeof(int32_t)) ||
                                 args.query_token_count != 1 || !args.is_causal)),
                           false, ERROR, "Invalid attention operation");
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
    if (call->has_nvidia_mma && call->type != backend::ScalarType::kFloat32 &&
        args.query_token_count > 1 && args.head_dim <= 128) {
      const size_t query_tile = args.head_dim <= 80 ? 128U : 64U;
      call->entry = "AttentionTensor";
      call->dimensions = 2U;
      call->local_size = {query_tile * 2U, 1U, 1U};
      call->global_size = {(static_cast<size_t>(args.query_token_count) + query_tile - 1U) /
                               query_tile * query_tile * 2U,
                           static_cast<size_t>(args.query_head_count), 1U};
      call->build_defines.push_back(args.head_dim <= 80 ? "OMNI_ATTENTION_DIM=80"
                                                        : "OMNI_ATTENTION_DIM=128");
    }
    if (args.query_token_count == 1 && args.is_causal &&
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
    return true;
  }

  static bool Run(const backend::OpenCLInvocation &invocation,
                  backend::OpenCLExecutor *const executor) {
    OMNI_RETURN_VAL_IF(executor == nullptr || !executor->Launch(invocation), false);
    if (invocation.entry == "AttentionDecode" || invocation.entry == "AttentionDecodeGrouped") {
      backend::OpenCLInvocation merge = invocation;
      merge.entry = "AttentionDecodeMerge";
      merge.dimensions = 1U;
      merge.local_size = {256U, 1U, 1U};
      merge.global_size = {static_cast<size_t>(merge.tensor_shape[1]) * 256U, 1U, 1U};
      return executor->Launch(merge);
    }
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
