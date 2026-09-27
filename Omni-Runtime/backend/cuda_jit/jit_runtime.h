#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <vector_types.h>

#include "Omni-Runtime/backend/cuda_jit/backend.h"
#include "Omni-Runtime/kernel/runtime/runtime.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace kernel {
namespace jit {

using CudaKernel = ::omni_runtime::jit::backend::cuda::Kernel;

class JitRuntime final {
public:
  class LaunchScope final {
  public:
    explicit LaunchScope(const void *const stream, const bool is_preparing,
                         std::vector<std::shared_ptr<CudaKernel>> *const kernels);
    ~LaunchScope();
    LaunchScope(const LaunchScope &) = delete;
    LaunchScope &operator=(const LaunchScope &) = delete;

  private:
    const void *previous_stream_;
    bool was_preparing_;
    std::vector<std::shared_ptr<CudaKernel>> *previous_kernels_;
  };

  static bool Init(const std::filesystem::path &library_root);
  static bool HasTensorMma();
  static bool HasSm120();
  static int32_t MultiprocessorCount();
  static const void *stream() {
    return stream_;
  }
  static bool is_preparing() {
    return is_preparing_;
  }
  static bool Compile(const std::string &tag, const std::string &source,
                      std::shared_ptr<CudaKernel> *const kernel);

  template <typename... Args>
  static bool Launch(const std::shared_ptr<CudaKernel> &kernel, const dim3 &grid_dim,
                     const dim3 &block_dim, const int shared_memory_bytes, const Args &...args) {
    OMNI_RETURN_VAL_IF(kernel == nullptr || shared_memory_bytes < 0, false);
    if (kernels_ != nullptr) {
      kernels_->push_back(kernel);
    }
    if (is_preparing_) {
      return true;
    }
    ::omni_runtime::jit::backend::cuda::LaunchOptions options;
    options.grid_dim = grid_dim;
    options.block_dim = block_dim;
    options.num_smem_bytes = shared_memory_bytes;
    options.stream = reinterpret_cast<CUstream>(const_cast<void *>(stream_));
    try {
      Runtime().launch(kernel, options, args...);
    } catch (...) {
      return false;
    }
    return true;
  }

private:
  static thread_local const void *stream_;
  static thread_local bool is_preparing_;
  static thread_local std::vector<std::shared_ptr<CudaKernel>> *kernels_;
  static ::omni_runtime::jit::backend::runtime::Runtime<::omni_runtime::jit::backend::cuda::CUDA> &
  Runtime();
  static ::omni_runtime::jit::utils::lazy::LazyInit<
      ::omni_runtime::jit::backend::runtime::Runtime<::omni_runtime::jit::backend::cuda::CUDA>> &
  LazyRuntime();
};

} // namespace jit
} // namespace kernel
} // namespace omni_runtime
