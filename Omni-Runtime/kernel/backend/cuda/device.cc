#include "Omni-Runtime/kernel/backend/cuda/device.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace cuda {

const cudaDeviceProp &Device::get_prop() {
  if (not initialized) {
    // `cudaFree(nullptr)` is to ensure the current CUDA context exists before later driver API
    // calls
    int device_index = 0;
    JIT_CUDA_DEVICE_CHECK(cudaGetDevice(&device_index));
    JIT_CUDA_DEVICE_CHECK(cudaFree(nullptr));
    JIT_CUDA_DEVICE_CHECK(cudaGetDeviceProperties(&prop, device_index));
    initialized = true;
  }
  return prop;
}

int Device::get_num_sms() {
  return get_prop().multiProcessorCount;
}

int Device::get_num_l2_cache_bytes() {
  return get_prop().l2CacheSize;
}

int Device::get_num_smem_bytes() {
  return static_cast<int>(get_prop().sharedMemPerBlockOptin);
}

int64_t Device::get_clock_rate() {
  if (clock_rate == 0) {
    int device_index = 0;
    int rate = 0;
    JIT_CUDA_DEVICE_CHECK(cudaGetDevice(&device_index));
    JIT_CUDA_DEVICE_CHECK(cudaDeviceGetAttribute(&rate, cudaDevAttrClockRate, device_index));
    clock_rate = static_cast<int64_t>(rate) * 1000;
  }
  return clock_rate;
}

int Device::get_arch_major() {
  return get_prop().major;
}

int Device::get_arch_minor() {
  return get_prop().minor;
}

std::pair<int, int> Device::get_arch_pair() {
  return {get_arch_major(), get_arch_minor()};
}

std::string Device::get_arch(const bool use_arch_family, const bool number_only) {
  const auto [major, minor] = get_arch_pair();
  const auto arch = std::to_string(major * 10 + minor);

  // E.g., sm_80, sm_90, sm_100
  if (number_only or major < 9) {
    return arch;
  }

  // E.g., sm_90a, sm_100f, sm_100a
  if (major > 9) {
    return arch + (use_arch_family ? "f" : "a");
  }
  return arch + "a";
}

} // namespace cuda
} // namespace backend
} // namespace jit
} // namespace omni_runtime
