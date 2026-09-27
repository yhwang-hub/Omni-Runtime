#pragma once

#include "kernels/operator_runtime.h"

#include "kernels/gqa_attention_kernel/attention_define.h"
#include "kernels/kernel_runtime.h"

namespace omni_runtime::kernels::runtime {

template <typename T, int32_t kStages, int32_t kSmallKeys, int32_t kLargeKeys>
bool LaunchMmaAttention(const std::string &tag, const AttentionArgs &args,
                        const std::string &include_path, const std::string &kernel_name) {
  OMNI_RETURN_VAL_IF(!IsValidAttention(args), false);
  if constexpr (sizeof(T) != 2U || std::is_integral_v<T>) {
    LOG_ERROR("MMA attention requires FP16 or BF16 input");
    return false;
  } else {
    OMNI_RETURN_VAL_IF_LOG(args.head_dim > 128 || !CudaKernelRuntime::HasTensorMma(), false, ERROR,
                           "MMA attention requires SM80+ and head dimension <= 128");
    const uint32_t query_tile = args.head_dim <= 80 ? 128U : 64U;
    static_assert(kStages == 1 || kStages == 2);
    const std::string source =
        args.head_dim <= 80
            ? CudaKernelRuntime::MakeKernelSource<T, 80, kSmallKeys, kStages>(include_path,
                                                                              kernel_name)
            : CudaKernelRuntime::MakeKernelSource<T, 128, kLargeKeys, kStages>(include_path,
                                                                               kernel_name);
    const int32_t dimensions = args.head_dim <= 80 ? 80 : 128;
    const int32_t keys = args.head_dim <= 80 ? kSmallKeys : kLargeKeys;
    const int32_t stage_bytes = 2 * kStages * keys * (dimensions + 8) * sizeof(T);
    const int32_t shared_bytes = stage_bytes > 48 * 1024 ? stage_bytes : 0;
    return CudaKernelRuntime::CompileAndLaunch(
        tag, source,
        {.grid_dim = {(static_cast<uint32_t>(args.query_token_count) + query_tile - 1U) /
                          query_tile,
                      static_cast<uint32_t>(args.query_head_count), 1U},
         .block_dim = {query_tile * 2U, 1U, 1U},
         .shared_memory_bytes = shared_bytes},
        static_cast<const T *>(args.query), static_cast<const T *>(args.key_cache),
        static_cast<const T *>(args.value_cache), args.query_token_count, args.query_head_count,
        args.key_value_head_count, args.cached_token_count, args.start_position, args.head_dim,
        args.is_causal, static_cast<T *>(args.output));
  }
}

} // namespace omni_runtime::kernels::runtime
