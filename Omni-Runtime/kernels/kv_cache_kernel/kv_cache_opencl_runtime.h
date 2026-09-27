#pragma once

#include <tuple>

#include "Omni-Runtime/backend/backend_define.h"

#include "Omni-Runtime/backend/backend_opencl_utils.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <> struct OpenCLOperator<backend::StoreKvCache> final {
  static bool Make(const backend::StoreKvCache &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    const uint64_t count = uint64_t(args.token_count) * args.key_value_size;
    const uint64_t capacity = uint64_t(args.cache_capacity) * args.key_value_size;
    const size_t bytes = backend::ElementBytes(call->type);
    const size_t cache_bytes = args.is_fp16_key_value ? sizeof(uint16_t) : bytes;
    OMNI_RETURN_VAL_IF_LOG(
        args.token_count <= 0 || args.key_value_size <= 0 || args.cache_capacity <= 0 ||
            args.start_position < 0 || !backend::HasElements(args.key, count, bytes) ||
            !backend::HasElements(args.value, count, bytes) ||
            !backend::HasElements(args.key_cache, capacity, cache_bytes) ||
            !backend::HasElements(args.value_cache, capacity, cache_bytes) ||
            (args.is_fp16_key_value && call->type != backend::ScalarType::kFloat32) ||
            (args.cache_position.IsValid() &&
             !backend::HasElements(args.cache_position, 1U, sizeof(int32_t))),
        false, ERROR, "Invalid KV cache operation");
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::ConfigureOpenCL(
                           "kv_cache_kernel/kv_cache_kernel.cl", "StoreKvCache",
                           (count + 255U) / 256U, 256U, call),
                       false);
    OMNI_RETURN_VAL_IF(
        !backend::opencl_internal::BindOpenCLArguments(
            std::forward_as_tuple(
                args.key, args.value, args.token_count, args.key_value_size, args.start_position,
                args.cache_capacity, args.key_cache, args.value_cache, args.cache_position,
                args.cache_position.IsValid(), args.is_fp16_cache, args.is_fp16_key_value),
            call),
        false);
    return true;
  }
};

template <> struct OpenCLOperator<backend::AdvanceSequence> final {
  static bool Make(const backend::AdvanceSequence &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF_LOG(!backend::HasElements(args.positions, 3U, sizeof(int32_t)) ||
                               !backend::HasElements(args.cache_position, 1U, sizeof(int32_t)),
                           false, ERROR, "Invalid sequence state");
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::ConfigureOpenCL(
                           "kv_cache_kernel/kv_cache_kernel.cl", "AdvanceSequence", 1U, 1U, call),
                       false);
    OMNI_RETURN_VAL_IF(!backend::opencl_internal::BindOpenCLArguments(
                           std::forward_as_tuple(args.positions, args.cache_position), call),
                       false);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
