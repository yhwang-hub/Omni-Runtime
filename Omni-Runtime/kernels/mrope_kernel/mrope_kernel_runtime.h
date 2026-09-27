#pragma once

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

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
