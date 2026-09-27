#pragma once

#include "kernels/operator_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::StoreKvCache> final {
  static bool Make(const backend::StoreKvCache &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const uint64_t count = uint64_t(args.token_count) * args.key_value_size;
    const uint64_t capacity = uint64_t(args.cache_capacity) * args.key_value_size;
    const size_t bytes = ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.key_value_size <= 0 ||
                               args.cache_capacity <= 0 || args.start_position < 0 ||
                               !HasElements(args.key, count, bytes) ||
                               !HasElements(args.value, count, bytes) ||
                               !HasElements(args.key_cache, capacity, bytes) ||
                               !HasElements(args.value_cache, capacity, bytes) ||
                               (args.cache_position.IsValid() &&
                                !HasElements(args.cache_position, 1U, sizeof(int32_t))),
                           false, ERROR, "Invalid KV cache operation");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("kv_cache_kernel/kv_cache_kernel.cl", "StoreKvCache",
                                        (count + 255U) / 256U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.key, args.value, args.token_count, args.key_value_size,
                        args.start_position, args.cache_capacity, args.key_cache, args.value_cache,
                        args.cache_position, args.cache_position.IsValid());
    return true;
  }
};

template <> struct OpenCLOperator<backend::AdvanceSequence> final {
  static bool Make(const backend::AdvanceSequence &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF_LOG(!HasElements(args.positions, 3U, sizeof(int32_t)) ||
                               !HasElements(args.cache_position, 1U, sizeof(int32_t)),
                           false, ERROR, "Invalid sequence state");
    OMNI_RETURN_VAL_IF(
        !ConfigureOpenCL("kv_cache_kernel/kv_cache_kernel.cl", "AdvanceSequence", 1U, 1U, call),
        false);
    BindOpenCLArguments(call, args.positions, args.cache_position);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
