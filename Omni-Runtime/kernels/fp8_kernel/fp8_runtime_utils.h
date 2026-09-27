#pragma once

#include "Omni-Runtime/backend/backend_compute_define.h"

namespace omni_runtime::kernels::runtime {

inline bool ValidateFp8(const backend::Fp8Linear &args, const size_t scalar_bytes) {
  return (scalar_bytes == sizeof(float) || scalar_bytes == sizeof(uint16_t)) && args.rows > 0 &&
         args.inner > 0 && args.columns > 0 && args.inner % 128 == 0 && args.columns % 128 == 0 &&
         args.block_reduction_groups > 0 && (args.inner / 128) % args.block_reduction_groups == 0 &&
         backend::HasElements(args.input, static_cast<uint64_t>(args.rows) * args.inner,
                              scalar_bytes) &&
         backend::HasElements(args.weight, static_cast<uint64_t>(args.columns) * args.inner, 1U) &&
         backend::HasElements(args.weight_scales,
                              static_cast<uint64_t>(args.columns / 128) * (args.inner / 128),
                              sizeof(float)) &&
         backend::HasElements(args.output, static_cast<uint64_t>(args.rows) * args.columns,
                              scalar_bytes);
}

} // namespace omni_runtime::kernels::runtime
