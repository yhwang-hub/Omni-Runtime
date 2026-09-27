#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/operator_runtime_common.h"

namespace omni_runtime::kernels::runtime {

inline int32_t GgmlGemvWarps(const backend::QuantizedLinear &args) {
  if (args.is_ggml_split_k && args.weight_type == 8 && args.rows == 1) {
    const int64_t iterations = (static_cast<int64_t>(args.inner) + 1023) / 1024;
    const int64_t wide_iterations = (iterations + 1) / 2;
    if ((wide_iterations * 2 - iterations) * 8 <= wide_iterations * 2) {
      return 8;
    }
  }
  return args.rows <= 4 ? 4 : 2;
}

inline int32_t GgmlTokenTile(const int32_t rows) {
  const int32_t tile_count = (rows - 1) / 128 + 1;
  const int32_t needed = (rows - 1) / tile_count + 1;
  for (const int32_t tile : {8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128}) {
    if (tile >= needed) {
      return tile;
    }
  }
  return 128;
}

inline int32_t GgmlStreamGroups(const backend::QuantizedLinear &args, const int32_t compute_units) {
  const int64_t tiles = static_cast<int64_t>((args.rows - 1) / GgmlTokenTile(args.rows) + 1) *
                        ((args.columns - 1) / 128 + 1);
  const int64_t waves = (tiles - 1) / compute_units + 1;
  return 100 * tiles / (compute_units * waves) >= 90
             ? static_cast<int32_t>(tiles)
             : static_cast<int32_t>(std::min<int64_t>(
                   compute_units,
                   tiles * ((args.inner - 1) / (args.weight_type == 40 ? 512 : 256) + 1)));
}

inline bool NeedGgmlSplitK(const backend::QuantizedLinear &args, const int32_t compute_units) {
  const bool has_q8_input = args.weight_type == 8 && args.activation_block_size == 32;
  const bool has_fp4_input =
      args.weight_type == 40 && args.is_packed_nvfp4 && args.activation_block_size == 16;
  if (!args.is_ggml_split_k || (!has_q8_input && !has_fp4_input) || args.rows <= 8 ||
      !args.activation_scales.IsValid() || compute_units <= 0) {
    return false;
  }
  const int64_t tiles = static_cast<int64_t>((args.rows - 1) / GgmlTokenTile(args.rows) + 1) *
                        ((args.columns - 1) / 128 + 1);
  return tiles % GgmlStreamGroups(args, compute_units) != 0;
}

inline bool HasAtMostTwoGgmlParts(const backend::QuantizedLinear &args,
                                  const int32_t compute_units) {
  const int32_t blocks = args.inner / (args.weight_type == 40 ? 64 : 32);
  const int64_t tiles = static_cast<int64_t>((args.rows - 1) / GgmlTokenTile(args.rows) + 1) *
                        ((args.columns - 1) / 128 + 1);
  // Aligned partitions are at least one complete tile long.
  return blocks % 8 == 0 && tiles >= GgmlStreamGroups(args, compute_units);
}

inline bool HasGgufWeight(const backend::DeviceView &weight, const int32_t type,
                          const int32_t inner, const int32_t columns) {
  OMNI_RETURN_VAL_IF(inner <= 0 || columns <= 0, false);
  const uint64_t count = static_cast<uint64_t>(inner) * columns;
  size_t bytes = 0U;
  size_t alignment = 1U;
  switch (type) {
  case 0:
    bytes = count * sizeof(float);
    alignment = sizeof(float);
    break;
  case 1:
  case 30:
    bytes = count * sizeof(uint16_t);
    alignment = sizeof(uint16_t);
    break;
  case 8:
    OMNI_RETURN_VAL_IF(inner % 32 != 0, false);
    bytes = count / 32U * 34U;
    alignment = sizeof(uint16_t);
    break;
  case 40:
    OMNI_RETURN_VAL_IF(inner % 64 != 0, false);
    bytes = count / 64U * 36U;
    break;
  default:
    return false;
  }
  return HasElements(weight, bytes, 1U) && weight.offset_bytes % alignment == 0U;
}

inline bool ValidateQuantized(const backend::QuantizedLinear &args, const size_t bytes) {
  const bool has_packed_input = args.is_packed_nvfp4;
  const bool has_valid_scales =
      !args.activation_scales.IsValid()
          ? args.activation_scales.data == nullptr && args.activation_scales.size_bytes == 0U
          : (args.weight_type == 8 || args.weight_type == 40) &&
                (args.activation_block_size == 16 || args.activation_block_size == 32) &&
                args.inner > 0 && args.inner % args.activation_block_size == 0 &&
                HasElements(args.activation_scales,
                            static_cast<uint64_t>(args.rows) *
                                (args.inner / args.activation_block_size),
                            sizeof(float));
  return (!has_packed_input ||
          (bytes == sizeof(float) && args.weight_type == 40 && args.activation_block_size == 16 &&
           args.activation_scales.IsValid())) &&
         (!args.is_packed_q8 ||
          (!has_packed_input && (args.weight_type == 8 || args.weight_type == 40) &&
           args.activation_block_size == 32 && args.activation_scales.IsValid())) &&
         has_valid_scales && args.rows > 0 && std::isfinite(args.weight_scale) &&
         HasGgufWeight(args.weight, args.weight_type, args.inner, args.columns) &&
         (has_packed_input
              ? HasElements(args.input, static_cast<uint64_t>(args.rows) * args.inner / 64U * 36U,
                            1U)
              : HasElements(args.input, static_cast<uint64_t>(args.rows) * args.inner,
                            args.is_packed_q8 ? 1U : bytes)) &&
         HasElements(args.output, static_cast<uint64_t>(args.rows) * args.columns, bytes);
}

inline bool ValidateQuantized(const backend::QuantizedEmbedding &args, const size_t bytes) {
  return args.token_count > 0 && HasElements(args.indices, args.token_count, sizeof(int32_t)) &&
         HasGgufWeight(args.weight, args.weight_type, args.hidden_size, args.vocabulary_size) &&
         HasElements(args.output, static_cast<uint64_t>(args.token_count) * args.hidden_size,
                     bytes);
}

inline bool ValidateQuantized(const backend::QuantizeActivation &args, const size_t bytes) {
  const int32_t block = args.weight_type == 8 || args.is_decode ? 32 : 16;
  return (!args.is_packed_nvfp4 || (bytes == sizeof(float) && args.weight_type == 40 &&
                                    !args.is_decode && args.columns % 64 == 0)) &&
         (!args.is_packed_q8 ||
          (!args.is_packed_nvfp4 && (args.weight_type == 8 || args.is_decode))) &&
         args.rows > 0 && args.columns > 0 && args.columns % 32 == 0 &&
         (args.weight_type == 8 || args.weight_type == 40) &&
         HasElements(args.input, static_cast<uint64_t>(args.rows) * args.columns, bytes) &&
         (args.is_packed_nvfp4
              ? HasElements(args.output,
                            static_cast<uint64_t>(args.rows) * args.columns / 64U * 36U, 1U)
              : HasElements(args.output, static_cast<uint64_t>(args.rows) * args.columns,
                            args.is_packed_q8 ? 1U : bytes)) &&
         HasElements(args.scales, static_cast<uint64_t>(args.rows) * (args.columns / block),
                     sizeof(float));
}

inline uint32_t QuantizeColumnGroups(const backend::QuantizeActivation &args) {
  const bool has_block_scales = args.weight_type == 8 || args.is_decode;
  return has_block_scales && args.rows <= 8
             ? std::min(64U, (static_cast<uint32_t>(args.columns) - 1U) / 256U + 1U)
             : 1U;
}

} // namespace omni_runtime::kernels::runtime
