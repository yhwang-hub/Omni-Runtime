#pragma once

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
        static_cast<T *>(args.key_cache), static_cast<T *>(args.value_cache));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
