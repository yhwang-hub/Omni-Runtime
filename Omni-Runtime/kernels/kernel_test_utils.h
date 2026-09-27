#pragma once

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector_types.h>

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/kernel/runtime/jit.h"
#include "Omni-Runtime/memory/buffer.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace kernels {
namespace test {

using Kernel = ::omni_runtime::kernel::jit::CudaKernel;

bool InitJit();
bool CompileKernel(const std::string &tag, const std::string &source,
                   std::shared_ptr<Kernel> *const kernel);
template <typename... Args>
bool LaunchKernel(const std::shared_ptr<Kernel> &kernel, const dim3 grid_dim, const dim3 block_dim,
                  const int shared_memory_bytes, const Args &...args) {
  return ::omni_runtime::kernel::jit::JitRuntime::Launch(kernel, grid_dim, block_dim,
                                                         shared_memory_bytes, args...);
}
bool SyncDevice();

class KernelTestFixture : public ::testing::Test {
public:
  KernelTestFixture() = default;
  ~KernelTestFixture() override = default;

  KernelTestFixture(const KernelTestFixture &) = delete;
  KernelTestFixture &operator=(const KernelTestFixture &) = delete;

protected:
  void SetUp() override {
    cuda_backend_ = backend::CreateBackend(backend::BackendKind::kCuda);
    if (cuda_backend_ == nullptr || !cuda_backend_->IsAvailable()) {
      GTEST_SKIP() << "CUDA backend is unavailable";
    }
    is_jit_initialized_ = InitJit();
  }

  bool Compile(const std::string &tag, const std::string &source) {
    OMNI_RETURN_VAL_IF(!is_jit_initialized_, false);
    OMNI_RETURN_VAL_IF(!CompileKernel(tag, source, &kernel_), false);
    return kernel_ != nullptr;
  }

  template <typename... Args>
  bool Launch(const dim3 grid_dim, const dim3 block_dim, const int shared_memory_bytes,
              const Args &...args) {
    return LaunchKernel(kernel_, grid_dim, block_dim, shared_memory_bytes, args...);
  }

  bool Sync() const {
    return SyncDevice();
  }

  const std::shared_ptr<backend::Backend> &backend() const {
    return cuda_backend_;
  }

  const std::shared_ptr<Kernel> &kernel() const {
    return kernel_;
  }

  bool is_jit_initialized() const {
    return is_jit_initialized_;
  }

private:
  std::shared_ptr<backend::Backend> cuda_backend_;
  std::shared_ptr<Kernel> kernel_;
  bool is_jit_initialized_ = false;
};

bool ReferenceAttention(const std::vector<float> &query, const std::vector<float> &key,
                        const std::vector<float> &value, const int32_t head_count,
                        const int32_t token_count, const int32_t head_dim, const bool is_causal,
                        std::vector<float> *const output);

template <typename T>
bool IsClose(const std::vector<T> &actual, const std::vector<T> &expected,
             const float absolute_tolerance) {
  if (actual.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float actual_value = static_cast<float>(actual[index]);
    const float expected_value = static_cast<float>(expected[index]);
    if (!std::isfinite(actual_value) ||
        std::fabs(actual_value - expected_value) > absolute_tolerance) {
      return false;
    }
  }
  return true;
}

} // namespace test
} // namespace kernels
} // namespace omni_runtime
