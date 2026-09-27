#pragma once

#include <cstdint>

namespace omni_runtime {
namespace kernels {

struct Float16 final {
  uint16_t bits;
};

struct BFloat16 final {
  uint16_t bits;
};

} // namespace kernels
} // namespace omni_runtime
