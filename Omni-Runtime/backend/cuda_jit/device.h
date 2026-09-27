#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "Omni-Runtime/kernel/utils/exception.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace cuda {

inline void check_cuda_runtime(const cudaError_t error, const char *expression) {
  if (error != cudaSuccess) {
    JIT_PANIC("{} failed with CUDA error {} ({}): {}", expression, static_cast<int>(error),
              cudaGetErrorName(error), cudaGetErrorString(error));
  }
}

#ifndef JIT_CUDA_DEVICE_CHECK
#define JIT_CUDA_DEVICE_CHECK(expr)                                                                \
  ::omni_runtime::jit::backend::cuda::check_cuda_runtime((expr), #expr)
#endif

class Device {
  cudaDeviceProp prop{};
  int64_t clock_rate = 0;
  bool initialized = false;

public:
  const cudaDeviceProp &get_prop();

  int get_num_sms();

  int get_num_l2_cache_bytes();

  int get_num_smem_bytes();

  int64_t get_clock_rate();

  int get_arch_major();

  int get_arch_minor();

  std::pair<int, int> get_arch_pair();

  std::string get_arch(const bool use_arch_family, const bool number_only);
};

} // namespace cuda
} // namespace backend
} // namespace jit
} // namespace omni_runtime
