#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "Omni-Runtime/backend/backend_compute_define.h"

namespace omni_runtime::kernels::runtime {

inline size_t RoundWorkSize(const uint64_t elements, const size_t local_size) {
  return static_cast<size_t>((elements + local_size - 1U) / local_size) * local_size;
}

inline bool HasElements(const backend::DeviceView &view, const uint64_t count,
                        const size_t element_bytes) {
  return view.IsValid() && element_bytes != 0U && view.offset_bytes % element_bytes == 0U &&
         count <= view.size_bytes / element_bytes;
}

inline size_t ElementBytes(const backend::ScalarType type) {
  return type == backend::ScalarType::kFloat32 ? sizeof(float) : sizeof(uint16_t);
}

} // namespace omni_runtime::kernels::runtime
