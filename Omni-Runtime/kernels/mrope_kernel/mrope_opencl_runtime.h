#pragma once

#include "kernels/operator_runtime.h"

namespace omni_runtime::kernels::runtime {

inline bool IsValidRopeShape(const int32_t query_heads, const int32_t key_heads,
                             const int32_t dimension, const int32_t temporal, const int32_t height,
                             const int32_t width, const float base) {
  return query_heads > 0 && key_heads > 0 && key_heads <= query_heads &&
         query_heads % key_heads == 0 && dimension > 0 && dimension % 2 == 0 && temporal >= 0 &&
         height >= 0 && width >= 0 && temporal + height + width == dimension / 2 && base > 0.0F;
}

template <> struct OpenCLOperator<backend::MultiAxisRope> final {
  static bool Make(const backend::MultiAxisRope &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const uint64_t query_count = uint64_t(args.tokens) * args.query_heads * args.head_dim;
    const uint64_t key_count = uint64_t(args.tokens) * args.key_value_heads * args.head_dim;
    const size_t bytes = ElementBytes(call->type);
    OMNI_RETURN_VAL_IF_LOG(
        args.tokens <= 0 ||
            !IsValidRopeShape(args.query_heads, args.key_value_heads, args.head_dim,
                              args.temporal_section, args.height_section, args.width_section,
                              args.frequency_base) ||
            !HasElements(args.query, query_count, bytes) ||
            !HasElements(args.key, key_count, bytes) ||
            !HasElements(args.position_ids, uint64_t(args.tokens) * 3U, sizeof(int32_t)) ||
            !HasElements(args.query_output, query_count, bytes) ||
            !HasElements(args.key_output, key_count, bytes),
        false, ERROR, "Invalid rotary operation");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("mrope_kernel/mrope_kernel.cl", "MultiAxisRope",
                                        (query_count + 255U) / 256U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.query, args.key, args.position_ids, args.tokens,
                        args.query_heads, args.key_value_heads, args.head_dim,
                        args.temporal_section, args.height_section, args.width_section,
                        args.frequency_base, args.query_output, args.key_output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::NormalizeRopeCache> final {
  static bool Make(const backend::NormalizeRopeCache &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const size_t bytes = ElementBytes(call->type);
    const uint64_t query_count = uint64_t(args.query_heads) * args.head_dim;
    const uint64_t key_count = uint64_t(args.key_value_heads) * args.head_dim;
    OMNI_RETURN_VAL_IF_LOG(
        !IsValidRopeShape(args.query_heads, args.key_value_heads, args.head_dim,
                          args.temporal_section, args.height_section, args.width_section,
                          args.frequency_base) ||
            args.head_dim > 256 || args.cache_capacity <= 0 || args.epsilon <= 0.0F ||
            !HasElements(args.query, query_count, bytes) ||
            !HasElements(args.key, key_count, bytes) ||
            !HasElements(args.value, key_count, bytes) ||
            !HasElements(args.query_weight, args.head_dim, bytes) ||
            !HasElements(args.key_weight, args.head_dim, bytes) ||
            !HasElements(args.positions, 3U, sizeof(int32_t)) ||
            !HasElements(args.cache_position, 1U, sizeof(int32_t)) ||
            !HasElements(args.query_output, query_count, bytes) ||
            !HasElements(args.key_cache, uint64_t(args.cache_capacity) * key_count, bytes) ||
            !HasElements(args.value_cache, uint64_t(args.cache_capacity) * key_count, bytes),
        false, ERROR, "Invalid normalized rotary cache operation");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("mrope_kernel/mrope_kernel.cl", "NormalizeRopeCache",
                                        args.query_heads + args.key_value_heads, 256U, call),
                       false);
    BindOpenCLArguments(call, args.query, args.key, args.value, args.query_weight, args.key_weight,
                        args.positions, args.cache_position, args.query_heads, args.key_value_heads,
                        args.head_dim, args.cache_capacity, args.temporal_section,
                        args.height_section, args.width_section, args.frequency_base, args.epsilon,
                        args.query_output, args.key_cache, args.value_cache);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
