#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <format>
#include <memory>
#include <type_traits>

#include <cuda.h>

#include "Omni-Runtime/kernel/backend/cuda/driver.h"
#include "Omni-Runtime/kernel/backend/cuda/options.h"
#include "Omni-Runtime/kernel/utils/env.h"
#include "Omni-Runtime/kernel/utils/exception.h"
#include "Omni-Runtime/kernel/utils/gil.h"
#include "Omni-Runtime/kernel/utils/no_ref_ptr.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace cuda {

template <typename T> inline void *kernel_arg_pointer(const T &value) {
  if constexpr (std::is_base_of_v<::omni_runtime::jit::utils::NoRefPtr, std::decay_t<T>>) {
    return value.ptr;
  } else {
    return const_cast<void *>(static_cast<const void *>(&value));
  }
}

// Immutable CUDA kernel handles with shared ownership. Driver resources are
// unloaded when the last shared owner is destroyed.
class Kernel {
public:
  CUlibrary library_handle{};
  CUfunction kernel_handle{};

  Kernel(const CUlibrary &library_handle, const CUfunction &kernel_handle)
      : library_handle(library_handle), kernel_handle(kernel_handle) {
  }

  ~Kernel() {
    unload();
  }
  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;
  Kernel(Kernel &&) = delete;
  Kernel &operator=(Kernel &&) = delete;

  static std::shared_ptr<Kernel> load(const std::filesystem::path &dir,
                                      const ::omni_runtime::jit::utils::Env &env);

  template <typename... Args>
  void launch(const LaunchOptions &launch_options, const Args &...args) const {
    JIT_HOST_ASSERT(kernel_handle != nullptr, "kernel must be loaded before launch");
    JIT_HOST_ASSERT(launch_options.num_smem_bytes.has_value(),
                    "CUDA dynamic shared-memory size must be specified");
    JIT_HOST_ASSERT(launch_options.grid_dim.has_value(), "CUDA grid dimension must be specified");
    JIT_HOST_ASSERT(launch_options.block_dim.has_value(), "CUDA block dimension must be specified");
    JIT_HOST_ASSERT(launch_options.cluster_dim.has_value(),
                    "CUDA cluster dimension must be specified");
    JIT_HOST_ASSERT(launch_options.cooperative.has_value(),
                    "CUDA cooperative option must be specified");
    JIT_HOST_ASSERT(launch_options.enable_pdl.has_value(), "CUDA PDL option must be specified");
    JIT_HOST_ASSERT(launch_options.nonportable_cluster_size_allowed.has_value(),
                    "CUDA non-portable cluster option must be specified");
    JIT_HOST_ASSERT(*launch_options.num_smem_bytes >= 0,
                    "CUDA dynamic shared-memory size must not be negative");
    JIT_HOST_ASSERT(launch_options.grid_dim->x > 0 and launch_options.grid_dim->y > 0 and
                        launch_options.grid_dim->z > 0,
                    "CUDA grid dimensions must be positive");
    JIT_HOST_ASSERT(launch_options.block_dim->x > 0 and launch_options.block_dim->y > 0 and
                        launch_options.block_dim->z > 0,
                    "CUDA block dimensions must be positive");
    JIT_HOST_ASSERT(launch_options.cluster_dim->x > 0, "CUDA cluster dimension must be positive");
    JIT_HOST_ASSERT(launch_options.cluster_dim->y == 1 and launch_options.cluster_dim->z == 1,
                    "only one-dimensional CUDA clusters are supported");

    if (*launch_options.num_smem_bytes > 0) {
      JIT_CUDA_DRIVER_CHECK(driver::lazy_cuFuncSetAttribute(
          kernel_handle, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          *launch_options.num_smem_bytes));
    }
    if (*launch_options.nonportable_cluster_size_allowed) {
      JIT_CUDA_DRIVER_CHECK(driver::lazy_cuFuncSetAttribute(
          kernel_handle, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
    }

    std::array<CUlaunchAttribute, 3> attributes{};
    unsigned int num_attributes = 0;
    if (*launch_options.cooperative) {
      auto &attribute = attributes[num_attributes++];
      attribute.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
      attribute.value.cooperative = 1;
    }
    if (launch_options.cluster_dim->x > 1) {
      auto &attribute = attributes[num_attributes++];
      attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      attribute.value.clusterDim.x = launch_options.cluster_dim->x;
      attribute.value.clusterDim.y = 1;
      attribute.value.clusterDim.z = 1;
    }
    if (*launch_options.enable_pdl) {
      auto &attribute = attributes[num_attributes++];
      attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute.value.programmaticStreamSerializationAllowed = 1;
    }

    void *kernel_arg_ptrs[sizeof...(Args) + 1] = {kernel_arg_pointer(args)..., nullptr};
    CUlaunchConfig config{};
    config.gridDimX = launch_options.grid_dim->x;
    config.gridDimY = launch_options.grid_dim->y;
    config.gridDimZ = launch_options.grid_dim->z;
    config.blockDimX = launch_options.block_dim->x;
    config.blockDimY = launch_options.block_dim->y;
    config.blockDimZ = launch_options.block_dim->z;
    config.sharedMemBytes = *launch_options.num_smem_bytes;
    config.hStream = launch_options.stream ? *launch_options.stream : nullptr;
    config.attrs = num_attributes == 0 ? nullptr : attributes.data();
    config.numAttrs = num_attributes;
    JIT_CUDA_DRIVER_CHECK(driver::lazy_cuLaunchKernelEx(
        &config, kernel_handle, sizeof...(Args) == 0 ? nullptr : kernel_arg_ptrs, nullptr));
  }

  void unload() noexcept;
};

} // namespace cuda
} // namespace backend
} // namespace jit
} // namespace omni_runtime
