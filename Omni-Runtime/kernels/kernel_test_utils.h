#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#include <bit>
#include <type_traits>

#include <gtest/gtest.h>

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/memory/buffer.h"
#include "kernels/kernel_define.h"

namespace omni_runtime {
namespace kernels {
namespace test {

bool InitJit();
bool SyncDevice();

template <typename T> T LowPrecision(const float value) {
  if constexpr (std::is_same_v<T, Float16>) {
    return {std::bit_cast<uint16_t>(static_cast<_Float16>(value))};
  } else {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return {static_cast<uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U)};
  }
}

template <typename T> float HostFloat(const T value) {
  if constexpr (std::is_same_v<T, Float16>) {
    return static_cast<float>(std::bit_cast<_Float16>(value.bits));
  } else {
    return std::bit_cast<float>(static_cast<uint32_t>(value.bits) << 16U);
  }
}

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

  bool Sync() const {
    return SyncDevice();
  }

  const std::shared_ptr<backend::Backend> &backend() const {
    return cuda_backend_;
  }

  bool is_jit_initialized() const {
    return is_jit_initialized_;
  }

private:
  std::shared_ptr<backend::Backend> cuda_backend_;
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
