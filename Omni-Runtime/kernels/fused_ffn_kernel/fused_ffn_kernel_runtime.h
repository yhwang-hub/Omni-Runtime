#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class FusedFfnKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const fc1_weight;
    const void *const fc1_bias;
    const void *const fc2_weight;
    const void *const fc2_bias;
    int32_t rows;
    int32_t input_size;
    int32_t intermediate_size;
    int32_t output_size;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.fc1_weight == nullptr ||
                           args.fc1_bias == nullptr || args.fc2_weight == nullptr ||
                           args.fc2_bias == nullptr || args.output == nullptr,
                       false);
    const int64_t shared_bytes =
        (static_cast<int64_t>(args.input_size) + args.intermediate_size) * sizeof(float);
    OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.input_size <= 0 || args.intermediate_size <= 0 ||
                               args.output_size <= 0 ||
                               shared_bytes > std::numeric_limits<int32_t>::max(),
                           false, ERROR, "Invalid fused FFN dimensions or shared memory size");
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/fused_ffn_kernel/fused_ffn_kernel.cuh",
            "omni_runtime::kernels::fused_ffn_kernel::FusedFfnImpl"),
        {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = static_cast<int32_t>(shared_bytes)},
        static_cast<const T *>(args.input), static_cast<const T *>(args.fc1_weight),
        static_cast<const T *>(args.fc1_bias), static_cast<const T *>(args.fc2_weight),
        static_cast<const T *>(args.fc2_bias), args.rows, args.input_size, args.intermediate_size,
        args.output_size, static_cast<T *>(args.output));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
