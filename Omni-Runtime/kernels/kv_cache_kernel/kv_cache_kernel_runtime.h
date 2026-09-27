#pragma once

#include "kernels/operator_runtime.h"

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class StoreKvCacheKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const key;
    const void *const value;
    int32_t token_count;
    int32_t key_value_size;
    int32_t start_position;
    int32_t cache_capacity;
    void *const key_cache;
    void *const value_cache;
    const int32_t *cache_position = nullptr;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.key == nullptr || args.value == nullptr || args.key_cache == nullptr ||
                           args.value_cache == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/kv_cache_kernel/kv_cache_kernel.cuh",
            "omni_runtime::kernels::kv_cache_kernel::StoreKvCacheImpl"),
        args.options, static_cast<const T *>(args.key), static_cast<const T *>(args.value),
        args.token_count, args.key_value_size, args.start_position, args.cache_capacity,
        static_cast<T *>(args.key_cache), static_cast<T *>(args.value_cache), args.cache_position);
  }
};

template <typename T> struct CudaOperator<T, backend::AdvanceSequence> final {
  static bool Launch(const backend::AdvanceSequence &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.positions.IsValid() || !args.cache_position.IsValid() ||
                               args.positions.backend_kind != backend::BackendKind::kCuda ||
                               args.cache_position.backend_kind != backend::BackendKind::kCuda,
                           false, ERROR, "Invalid sequence state");
    return CudaKernelRuntime::CompileAndLaunch(
        "backend_AdvanceSequence",
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/kv_cache_kernel/kv_cache_kernel.cuh",
            "omni_runtime::kernels::kv_cache_kernel::AdvanceSequenceImpl"),
        {.grid_dim = {1U, 1U, 1U}, .block_dim = {1U, 1U, 1U}, .shared_memory_bytes = 0},
        CudaAddress<int32_t>(args.positions), CudaAddress<int32_t>(args.cache_position));
  }
};

template <typename T> struct CudaOperator<T, backend::StoreKvCache> final {
  static bool Launch(const backend::StoreKvCache &args) {
    OMNI_RETURN_VAL_IF_LOG(
        !args.key.IsValid() || !args.value.IsValid() || !args.key_cache.IsValid() ||
            !args.value_cache.IsValid() ||
            ((!args.cache_position.IsValid() ||
              args.cache_position.backend_kind != backend::BackendKind::kCuda) &&
             (args.cache_position.data != nullptr || args.cache_position.offset_bytes != 0U ||
              args.cache_position.size_bytes != 0U)),
        false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.key_value_size <= 0 ||
                               args.cache_capacity <= 0,
                           false, ERROR, "Invalid compute operation dimensions");
    return StoreKvCacheKernelRuntime<T>::CompileAndLaunch(
        "backend_StoreKvCache",
        {.options = {.grid_dim = {static_cast<uint32_t>(
                                      (uint64_t(args.token_count) * args.key_value_size + 255U) /
                                      256U),
                                  1U, 1U},
                     .block_dim = {256U, 1U, 1U},
                     .shared_memory_bytes = 0},
         .key = CudaAddress<void>(args.key),
         .value = CudaAddress<void>(args.value),
         .token_count = args.token_count,
         .key_value_size = args.key_value_size,
         .start_position = args.start_position,
         .cache_capacity = args.cache_capacity,
         .key_cache = CudaAddress<void>(args.key_cache),
         .value_cache = CudaAddress<void>(args.value_cache),
         .cache_position = CudaAddress<const int32_t>(args.cache_position)});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
