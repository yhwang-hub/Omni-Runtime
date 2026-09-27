#pragma once

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

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
      const bool is_vectorized = args.inner % (16 / sizeof(T)) == 0 &&
                                 reinterpret_cast<uintptr_t>(args.lhs) % 16U == 0U &&
                                 reinterpret_cast<uintptr_t>(args.rhs) % 16U == 0U;
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
