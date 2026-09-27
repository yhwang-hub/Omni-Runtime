#pragma once

#include "kernels/operator_runtime.h"
#include "Omni-Runtime/backend/backend_compute_define.h"

#include <algorithm>

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class SplitGatedQueryKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const query_gate;
    int32_t token_count;
    int32_t head_count;
    int32_t head_size;
    void *const query;
    void *const gate;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.query_gate == nullptr || args.query == nullptr || args.gate == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/gated_attention_kernel/gated_attention_kernel.cuh",
            "omni_runtime::kernels::gated_attention_kernel::SplitGatedQueryImpl"),
        args.options, static_cast<const T *>(args.query_gate), args.token_count, args.head_count,
        args.head_size, static_cast<T *>(args.query), static_cast<T *>(args.gate));
  }
};

template <typename T> struct CudaOperator<T, backend::SplitGatedQuery> final {
  static bool Launch(const backend::SplitGatedQuery &args) {
    OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.head_count <= 0 || args.head_size <= 0,
                           false, ERROR, "Invalid CUDA gated query dimensions");
    const uint64_t rows = uint64_t(args.token_count) * args.head_count;
    OMNI_RETURN_VAL_IF_LOG(rows > uint64_t(std::numeric_limits<int64_t>::max()) /
                                      (2U * uint64_t(args.head_size)),
                           false, ERROR, "CUDA gated query dimensions overflow");
    const uint64_t count = rows * args.head_size;
    OMNI_RETURN_VAL_IF_LOG(!HasElements(args.query_gate, 2U * count, sizeof(T)) ||
                               !HasElements(args.query, count, sizeof(T)) ||
                               !HasElements(args.gate, count, sizeof(T)),
                           false, ERROR, "Invalid CUDA gated query buffers");
    const uint32_t groups =
        static_cast<uint32_t>(std::min<uint64_t>((count - 1U) / 256U + 1U, 65535U));
    return SplitGatedQueryKernelRuntime<T>::CompileAndLaunch(
        "backend_SplitGatedQuery", {.options = {.grid_dim = {groups, 1U, 1U},
                                                .block_dim = {256U, 1U, 1U},
                                                .shared_memory_bytes = 0},
                                    .query_gate = CudaAddress<void>(args.query_gate),
                                    .token_count = args.token_count,
                                    .head_count = args.head_count,
                                    .head_size = args.head_size,
                                    .query = CudaAddress<void>(args.query),
                                    .gate = CudaAddress<void>(args.gate)});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
