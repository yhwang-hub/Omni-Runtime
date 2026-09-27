#pragma once

#include "kernels/cuda_operator_runtime.h"

#include <cstdint>
#include <algorithm>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T>
bool IsVectorizedGemv(const void *const input, const void *const weight, const int32_t inner) {
  return inner % (16 / sizeof(T)) == 0 && reinterpret_cast<uintptr_t>(input) % 16U == 0U &&
         reinterpret_cast<uintptr_t>(weight) % 16U == 0U;
}

template <typename T, int32_t kInner> class BlackwellGemvKernelRuntime final {
  static_assert(kInner > 0 && kInner % (16 / sizeof(T)) == 0);

public:
  static bool CompileAndLaunch(const std::string &tag, const void *const input,
                               const void *const weight, const int32_t columns,
                               void *const output) {
    OMNI_RETURN_VAL_IF_LOG(input == nullptr || weight == nullptr || output == nullptr ||
                               columns <= 0 || reinterpret_cast<uintptr_t>(input) % 16U != 0U ||
                               reinterpret_cast<uintptr_t>(weight) % 16U != 0U,
                           false, ERROR, "Invalid aligned Blackwell GEMV inputs");
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        CudaKernelRuntime::MakeKernelSource<T, kInner>(
            "kernels/matmul_kernel/matmul_kernel.cuh",
            "omni_runtime::kernels::matmul_kernel::BlackwellGemvImpl"),
        {.grid_dim = {static_cast<uint32_t>((columns + 7) / 8), 1U, 1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = 0},
        static_cast<const T *>(input), static_cast<const T *>(weight), columns,
        static_cast<T *>(output));
  }
};

template <typename T> class ParallelGemvKernelRuntime final {
public:
  static bool CanCombine(const backend::ParallelLinear &args) {
    const auto &first = args.projections.front();
    const bool is_vectorized =
        IsVectorizedGemv<T>(CudaAddress<T>(first.input), CudaAddress<T>(first.weight), first.inner);
    return std::all_of(args.projections.begin(), args.projections.end(),
                       [is_vectorized](const auto &projection) {
                         return IsVectorizedGemv<T>(CudaAddress<T>(projection.input),
                                                    CudaAddress<T>(projection.weight),
                                                    projection.inner) == is_vectorized;
                       });
  }

  static bool CompileAndLaunch(const backend::ParallelLinear &args) {
    const auto &projections = args.projections;
    const auto &first = projections[0];
    const auto &second = projections[1];
    const bool has_third = projections.size() == 3U;
    const auto &third = has_third ? projections[2] : second;
    bool is_vectorized = first.inner % (16 / sizeof(T)) == 0 &&
                         reinterpret_cast<uintptr_t>(CudaAddress<T>(first.input)) % 16U == 0U;
    uint32_t blocks = 0U;
    for (const auto &projection : projections) {
      is_vectorized = is_vectorized &&
                      reinterpret_cast<uintptr_t>(CudaAddress<T>(projection.weight)) % 16U == 0U;
      blocks += (static_cast<uint32_t>(projection.columns) + 7U) / 8U;
    }
    constexpr const char *kSource = "kernels/matmul_kernel/matmul_kernel.cuh";
    constexpr const char *kEntry = "omni_runtime::kernels::matmul_kernel::ParallelGemvImpl";
    std::string source;
    if (is_vectorized && sizeof(T) == 2U && CudaKernelRuntime::HasSm120() &&
        (first.inner == 4096 || first.inner == 12288)) {
      source = first.inner == 4096
                   ? CudaKernelRuntime::MakeKernelSource<T, true, 4096>(kSource, kEntry)
                   : CudaKernelRuntime::MakeKernelSource<T, true, 12288>(kSource, kEntry);
    } else {
      source = is_vectorized ? CudaKernelRuntime::MakeKernelSource<T, true, 0>(kSource, kEntry)
                             : CudaKernelRuntime::MakeKernelSource<T, false, 0>(kSource, kEntry);
    }
    return CudaKernelRuntime::CompileAndLaunch(
        "backend_parallel_gemv", source,
        {.grid_dim = {blocks, 1U, 1U}, .block_dim = {256U, 1U, 1U}, .shared_memory_bytes = 0},
        CudaAddress<T>(first.input), CudaAddress<T>(first.weight), first.columns,
        CudaAddress<T>(first.output), CudaAddress<T>(second.weight), second.columns,
        CudaAddress<T>(second.output), CudaAddress<T>(third.weight), has_third ? third.columns : 0,
        CudaAddress<T>(third.output), first.inner);
  }
};

template <typename T> class MatmulKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const lhs;
    const void *const rhs;
    int32_t rows;
    int32_t inner;
    int32_t columns;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.lhs == nullptr || args.rhs == nullptr || args.output == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.inner <= 0 || args.columns <= 0, false, ERROR,
                           "Invalid matrix multiplication dimensions");
    const uint32_t tile_columns = args.rows <= 16 ? 16U : 64U;
    const std::string source = args.rows <= 16
                                   ? CudaKernelRuntime::MakeKernelSource<T, 16>(
                                         "kernels/matmul_kernel/matmul_kernel.cuh",
                                         "omni_runtime::kernels::matmul_kernel::MatmulImpl")
                                   : CudaKernelRuntime::MakeKernelSource<T, 64>(
                                         "kernels/matmul_kernel/matmul_kernel.cuh",
                                         "omni_runtime::kernels::matmul_kernel::MatmulImpl");
    return CudaKernelRuntime::CompileAndLaunch(
        tag, source,
        {.grid_dim = {(static_cast<uint32_t>(args.columns) + tile_columns - 1U) / tile_columns,
                      (static_cast<uint32_t>(args.rows) + 15U) / 16U, 1U},
         .block_dim = {16U, 16U, 1U},
         .shared_memory_bytes = 0},
        static_cast<const T *>(args.lhs), static_cast<const T *>(args.rhs), args.rows, args.inner,
        args.columns, static_cast<T *>(args.output));
  }
};

template <typename T> class LinearMatmulKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const lhs;
    const void *const rhs;
    int32_t rows;
    int32_t inner;
    int32_t columns;
    void *const output;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.lhs == nullptr || args.rhs == nullptr || args.output == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.inner <= 0 || args.columns <= 0, false, ERROR,
                           "Invalid linear projection dimensions");
    if (args.rows == 1) {
      const bool is_vectorized = IsVectorizedGemv<T>(args.lhs, args.rhs, args.inner);
      if constexpr (sizeof(T) == 2U) {
        if (is_vectorized && args.columns <= 12288 && CudaKernelRuntime::HasSm120()) {
          if (args.inner == 4096) {
            return BlackwellGemvKernelRuntime<T, 4096>::CompileAndLaunch(tag, args.lhs, args.rhs,
                                                                         args.columns, args.output);
          }
          if (args.inner == 12288) {
            return BlackwellGemvKernelRuntime<T, 12288>::CompileAndLaunch(
                tag, args.lhs, args.rhs, args.columns, args.output);
          }
        }
      }
      const std::string source = is_vectorized
                                     ? CudaKernelRuntime::MakeKernelSource<T, true>(
                                           "kernels/matmul_kernel/matmul_kernel.cuh",
                                           "omni_runtime::kernels::matmul_kernel::LinearGemvImpl")
                                     : CudaKernelRuntime::MakeKernelSource<T, false>(
                                           "kernels/matmul_kernel/matmul_kernel.cuh",
                                           "omni_runtime::kernels::matmul_kernel::LinearGemvImpl");
      return CudaKernelRuntime::CompileAndLaunch(
          tag, source,
          {.grid_dim = {static_cast<uint32_t>((args.columns + 7) / 8), 1U, 1U},
           .block_dim = {256U, 1U, 1U},
           .shared_memory_bytes = 0},
          static_cast<const T *>(args.lhs), static_cast<const T *>(args.rhs), args.inner,
          args.columns, static_cast<T *>(args.output));
    }
    const uint32_t tile_columns = args.rows <= 16 ? 16U : 64U;
    const std::string source = args.rows <= 16
                                   ? CudaKernelRuntime::MakeKernelSource<T, 16>(
                                         "kernels/matmul_kernel/matmul_kernel.cuh",
                                         "omni_runtime::kernels::matmul_kernel::LinearMatmulImpl")
                                   : CudaKernelRuntime::MakeKernelSource<T, 64>(
                                         "kernels/matmul_kernel/matmul_kernel.cuh",
                                         "omni_runtime::kernels::matmul_kernel::LinearMatmulImpl");
    return CudaKernelRuntime::CompileAndLaunch(
        tag, source,
        {.grid_dim = {(static_cast<uint32_t>(args.columns) + tile_columns - 1U) / tile_columns,
                      (static_cast<uint32_t>(args.rows) + 15U) / 16U, 1U},
         .block_dim = {16U, 16U, 1U},
         .shared_memory_bytes = 0},
        static_cast<const T *>(args.lhs), static_cast<const T *>(args.rhs), args.rows, args.inner,
        args.columns, static_cast<T *>(args.output));
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
