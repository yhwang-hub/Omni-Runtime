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

template <typename T> class GatedAddKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const lhs;
    const void *const rhs;
    const void *const gate;
    uint64_t element_count;
    int32_t hidden_size;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.lhs == nullptr || args.rhs == nullptr || args.gate == nullptr ||
                           args.output == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/gated_add_kernel/gated_add_kernel.cuh",
            "omni_runtime::kernels::gated_add_kernel::GatedAddImpl"),
        args.options, static_cast<const T *>(args.lhs), static_cast<const T *>(args.rhs),
        static_cast<const T *>(args.gate), args.element_count, args.hidden_size,
        static_cast<T *>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::GatedAdd> final {
  static bool Launch(const backend::GatedAdd &args) {
    OMNI_RETURN_VAL_IF_LOG(args.element_count == 0U || args.hidden_size <= 0 ||
                               !HasElements(args.lhs, args.element_count, sizeof(T)) ||
                               !HasElements(args.rhs, args.element_count, sizeof(T)) ||
                               !HasElements(args.gate, args.hidden_size, sizeof(T)) ||
                               !HasElements(args.output, args.element_count, sizeof(T)),
                           false, ERROR, "Invalid CUDA gated addition operation");
    const uint32_t groups =
        static_cast<uint32_t>(std::min<uint64_t>((args.element_count - 1U) / 256U + 1U, 65535U));
    return GatedAddKernelRuntime<T>::CompileAndLaunch("backend_GatedAdd",
                                                      {.options = {.grid_dim = {groups, 1U, 1U},
                                                                   .block_dim = {256U, 1U, 1U},
                                                                   .shared_memory_bytes = 0},
                                                       .lhs = CudaAddress<void>(args.lhs),
                                                       .rhs = CudaAddress<void>(args.rhs),
                                                       .gate = CudaAddress<void>(args.gate),
                                                       .element_count = args.element_count,
                                                       .hidden_size = args.hidden_size,
                                                       .output = CudaAddress<void>(args.output)});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
