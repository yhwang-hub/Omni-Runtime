#pragma once

#include <vector_types.h>
#include <cstdint>

namespace omni_runtime {
namespace kernels {

struct Float16 final {
  uint16_t bits;
};

struct BFloat16 final {
  uint16_t bits;
};

struct KernelLaunchConfig final {
  dim3 grid_dim;
  dim3 block_dim;
  int shared_memory_bytes;
};

} // namespace kernels
} // namespace omni_runtime
