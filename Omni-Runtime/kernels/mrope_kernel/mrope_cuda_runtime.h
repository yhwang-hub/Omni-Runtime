#pragma once

#include "kernels/cuda_operator_runtime.h"

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class MropeKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const query;
    const void *const key;
    const void *const position_ids;
    int32_t tokens;
    int32_t query_heads;
    int32_t key_value_heads;
    int32_t head_dim;
    int32_t temporal_section;
    int32_t height_section;
    int32_t width_section;
    float frequency_base;
    void *const query_output;
    void *const key_output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.query == nullptr || args.key == nullptr ||
                           args.position_ids == nullptr || args.query_output == nullptr ||
                           args.key_output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>("kernels/mrope_kernel/mrope_kernel.cuh",
                                               "omni_runtime::kernels::mrope_kernel::MropeImpl"),
        args.options, static_cast<const T *>(args.query), static_cast<const T *>(args.key),
        static_cast<const T *>(args.position_ids), args.tokens, args.query_heads,
        args.key_value_heads, args.head_dim, args.temporal_section, args.height_section,
        args.width_section, args.frequency_base, static_cast<T *>(args.query_output),
        static_cast<T *>(args.key_output));
  }
};

template <typename T> struct CudaOperator<T, backend::MultiAxisRope> final {
  static bool Launch(const backend::MultiAxisRope &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.query.IsValid() || !args.key.IsValid() ||
                               !args.position_ids.IsValid() || !args.query_output.IsValid() ||
                               !args.key_output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.tokens <= 0 || args.query_heads <= 0 || args.key_value_heads <= 0 ||
                               args.head_dim <= 0,
                           false, ERROR, "Invalid compute operation dimensions");
    return MropeKernelRuntime<T>::CompileAndLaunch(
        "backend_MultiAxisRope",
        {.options = {.grid_dim = {static_cast<uint32_t>(
                                      (uint64_t(args.tokens) * args.query_heads * args.head_dim +
                                       255U) /
                                      256U),
                                  1U, 1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .query = CudaAddress<void>(args.query),
         .key = CudaAddress<void>(args.key),
         .position_ids = CudaAddress<void>(args.position_ids),
         .tokens = args.tokens,
         .query_heads = args.query_heads,
         .key_value_heads = args.key_value_heads,
         .head_dim = args.head_dim,
         .temporal_section = args.temporal_section,
         .height_section = args.height_section,
         .width_section = args.width_section,
         .frequency_base = args.frequency_base,
         .query_output = CudaAddress<void>(args.query_output),
         .key_output = CudaAddress<void>(args.key_output)});
  }
};

template <typename T> struct CudaOperator<T, backend::NormalizeRopeCache> final {
  static bool Launch(const backend::NormalizeRopeCache &args) {
    OMNI_RETURN_VAL_IF_LOG(
        !args.query.IsValid() || !args.key.IsValid() || !args.value.IsValid() ||
            !args.query_weight.IsValid() || !args.key_weight.IsValid() ||
            !args.positions.IsValid() || !args.cache_position.IsValid() ||
            !args.query_output.IsValid() || !args.key_cache.IsValid() ||
            !args.value_cache.IsValid() || args.query.backend_kind != backend::BackendKind::kCuda ||
            args.key.backend_kind != backend::BackendKind::kCuda ||
            args.value.backend_kind != backend::BackendKind::kCuda ||
            args.query_weight.backend_kind != backend::BackendKind::kCuda ||
            args.key_weight.backend_kind != backend::BackendKind::kCuda ||
            args.positions.backend_kind != backend::BackendKind::kCuda ||
            args.cache_position.backend_kind != backend::BackendKind::kCuda ||
            args.query_output.backend_kind != backend::BackendKind::kCuda ||
            args.key_cache.backend_kind != backend::BackendKind::kCuda ||
            args.value_cache.backend_kind != backend::BackendKind::kCuda || args.query_heads <= 0 ||
            args.key_value_heads <= 0 || args.head_dim <= 0 || args.head_dim > 256 ||
            args.head_dim % 2 != 0 || args.cache_capacity <= 0 || args.temporal_section < 0 ||
            args.height_section < 0 || args.width_section < 0 ||
            args.temporal_section + args.height_section + args.width_section != args.head_dim / 2 ||
            args.frequency_base <= 0.0F,
        false, ERROR, "Invalid normalized rotary cache inputs");
    return CudaKernelRuntime::CompileAndLaunch(
        "backend_NormalizeRopeCache",
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/mrope_kernel/mrope_kernel.cuh",
            "omni_runtime::kernels::mrope_kernel::NormalizeRopeCacheImpl"),
        {.grid_dim = {static_cast<uint32_t>(args.query_heads + args.key_value_heads), 1U, 1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = 0},
        CudaAddress<const T>(args.query), CudaAddress<const T>(args.key),
        CudaAddress<const T>(args.value), CudaAddress<const T>(args.query_weight),
        CudaAddress<const T>(args.key_weight), CudaAddress<const int32_t>(args.positions),
        CudaAddress<const int32_t>(args.cache_position), args.query_heads, args.key_value_heads,
        args.head_dim, args.temporal_section, args.height_section, args.width_section,
        args.frequency_base, args.epsilon, CudaAddress<T>(args.query_output),
        CudaAddress<T>(args.key_cache), CudaAddress<T>(args.value_cache));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
