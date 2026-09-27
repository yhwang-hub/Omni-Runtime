#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/operator_runtime_common.h"

namespace omni_runtime::kernels::runtime {

inline bool ValidateDelta(const backend::CausalConv1d &args, const size_t bytes) {
  return args.token_count > 0 && args.channels > 0 && args.kernel_size >= 2 &&
         args.kernel_size <= 16 &&
         args.channels <= std::numeric_limits<int32_t>::max() / args.kernel_size &&
         HasElements(args.input, static_cast<uint64_t>(args.token_count) * args.channels, bytes) &&
         HasElements(args.output, static_cast<uint64_t>(args.token_count) * args.channels, bytes) &&
         HasElements(args.weight, static_cast<uint64_t>(args.channels) * args.kernel_size,
                     sizeof(float)) &&
         HasElements(args.state, static_cast<uint64_t>(args.channels) * (args.kernel_size - 1),
                     sizeof(float));
}

inline bool ValidateDelta(const backend::GatedDeltaNet &args, const size_t bytes) {
  OMNI_RETURN_VAL_IF(args.token_count <= 0 || args.key_head_count <= 0 ||
                         args.value_head_count <= 0 ||
                         args.value_head_count % args.key_head_count != 0 || args.head_size <= 0 ||
                         args.head_size > 128 || args.head_size % 8 != 0 ||
                         !std::isfinite(args.epsilon) || args.epsilon <= 0.0F,
                     false);
  const uint64_t channels =
      (2U * static_cast<uint64_t>(args.key_head_count) + args.value_head_count) * args.head_size;
  OMNI_RETURN_VAL_IF(channels > std::numeric_limits<int32_t>::max(), false);
  const uint64_t gates = static_cast<uint64_t>(args.token_count) * args.value_head_count;
  return HasElements(args.qkv, static_cast<uint64_t>(args.token_count) * channels, bytes) &&
         HasElements(args.alpha, gates, bytes) && HasElements(args.beta, gates, bytes) &&
         HasElements(args.decay_weight, args.value_head_count, sizeof(float)) &&
         HasElements(args.time_bias, args.value_head_count, sizeof(float)) &&
         HasElements(args.state,
                     static_cast<uint64_t>(args.value_head_count) * args.head_size * args.head_size,
                     sizeof(float)) &&
         HasElements(args.output, gates * args.head_size, bytes);
}

inline bool ValidateDelta(const backend::SigmoidGate &args, const size_t bytes) {
  return args.element_count > 0U && HasElements(args.input, args.element_count, bytes) &&
         HasElements(args.gate, args.element_count, bytes) &&
         HasElements(args.output, args.element_count, bytes);
}

inline bool ValidateDelta(const backend::PartialMultiAxisRope &args, const size_t bytes) {
  OMNI_RETURN_VAL_IF(
      args.tokens <= 0 || args.query_heads <= 0 || args.key_value_heads <= 0 ||
          args.head_dim <= 0 || args.rotary_dim <= 0 || args.rotary_dim > args.head_dim ||
          args.rotary_dim % 2 != 0 || args.temporal_section < 0 || args.height_section < 0 ||
          args.width_section < 0 ||
          static_cast<int64_t>(args.temporal_section) + args.height_section + args.width_section !=
              args.rotary_dim / 2 ||
          !std::isfinite(args.frequency_base) || args.frequency_base <= 0.0F,
      false);
  const uint64_t rows =
      static_cast<uint64_t>(args.tokens) * std::max(args.query_heads, args.key_value_heads);
  OMNI_RETURN_VAL_IF(rows > std::numeric_limits<uint64_t>::max() / args.head_dim, false);
  return HasElements(args.position_ids, static_cast<uint64_t>(args.tokens) * 3U, sizeof(int32_t)) &&
         HasElements(args.query,
                     static_cast<uint64_t>(args.tokens) * args.query_heads * args.head_dim,
                     bytes) &&
         HasElements(args.query_output,
                     static_cast<uint64_t>(args.tokens) * args.query_heads * args.head_dim,
                     bytes) &&
         HasElements(args.key,
                     static_cast<uint64_t>(args.tokens) * args.key_value_heads * args.head_dim,
                     bytes) &&
         HasElements(args.key_output,
                     static_cast<uint64_t>(args.tokens) * args.key_value_heads * args.head_dim,
                     bytes);
}

} // namespace omni_runtime::kernels::runtime
