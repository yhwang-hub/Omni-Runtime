#pragma once

#include <vector_types.h>

namespace omni_runtime::kernels {

struct KernelLaunchConfig final {
  dim3 grid_dim;
  dim3 block_dim;
  int shared_memory_bytes;
};

} // namespace omni_runtime::kernels
