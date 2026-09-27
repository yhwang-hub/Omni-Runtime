#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <vector_types.h>

#include "Omni-Runtime/kernel/backend/cuda/backend.h"
#include "Omni-Runtime/kernel/runtime/runtime.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace kernel {
namespace jit {

using CudaKernel = ::omni_runtime::jit::backend::cuda::Kernel;

class JitRuntime final {
public:
  static bool Init(const std::filesystem::path &library_root);
  static bool Compile(const std::string &tag, const std::string &source,
                      std::shared_ptr<CudaKernel> *const kernel);

  template <typename... Args>
  static bool Launch(const std::shared_ptr<CudaKernel> &kernel, const dim3 &grid_dim,
                     const dim3 &block_dim, const int shared_memory_bytes, const Args &...args) {
    OMNI_RETURN_VAL_IF(kernel == nullptr || shared_memory_bytes < 0, false);
    ::omni_runtime::jit::backend::cuda::LaunchOptions options;
    options.grid_dim = grid_dim;
    options.block_dim = block_dim;
    options.num_smem_bytes = shared_memory_bytes;
    try {
      Runtime().launch(kernel, options, args...);
    } catch (...) {
      return false;
    }
    return true;
  }

private:
  static ::omni_runtime::jit::backend::runtime::Runtime<::omni_runtime::jit::backend::cuda::CUDA> &
  Runtime();
  static ::omni_runtime::jit::utils::lazy::LazyInit<
      ::omni_runtime::jit::backend::runtime::Runtime<::omni_runtime::jit::backend::cuda::CUDA>> &
  LazyRuntime();
};

} // namespace jit
} // namespace kernel
} // namespace omni_runtime
