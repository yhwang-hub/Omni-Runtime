#pragma once

#include <kernels/common/kernel_type_utils.h>
#include <kernels/matmul_kernel/matmul_kernel.cuh>
#include <kernels/matmul_kernel/tensor_core.cuh>

namespace omni_runtime::kernels::fp8_kernel {

__device__ inline int32_t CodeInteger(const uint8_t code) {
  const uint32_t magnitude = code & 127U;
  const uint32_t exponent = magnitude >> 3U;
  const uint32_t mantissa = magnitude & 7U;
  const int32_t value = exponent == 0U ? mantissa : (8U + mantissa) << (exponent - 1U);
  return code & 128U ? -value : value;
}

__device__ inline int32_t IntegerDigit(const int32_t integer, const int32_t digit) {
  int32_t value = integer;
  for (int32_t index = 0; index < digit; ++index) {
    value = (value + 64) >> 7;
  }
  return digit == 2 ? value : ((value + 64) & 127) - 64;
}

__device__ inline uint8_t QuantizeCode(const float value) {
  uint16_t packed;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(packed) : "f"(value), "f"(0.0F));
  return static_cast<uint8_t>(packed);
}

template <typename T, int32_t kIsDecode>
__global__ void QuantizeImpl(const T *const input, const int32_t inner, uint8_t *const activation,
                             float *const scales) {
  const int32_t lane = threadIdx.x % 32;
  const int32_t block = blockIdx.x * 8 + threadIdx.x / 32;
  const int32_t row = blockIdx.y;
  const int32_t blocks = inner / 128;
  if (block >= blocks) {
    return;
  }
  const uint64_t first = static_cast<uint64_t>(row) * inner + block * 128;
  float values[4];
  float maximum = 1e-12F;
#pragma unroll
  for (int32_t part = 0; part < 4; ++part) {
    values[part] = ToFloat(input[first + part * 32 + lane]);
    maximum = fmaxf(maximum, fabsf(values[part]));
  }
#pragma unroll
  for (int32_t distance = 16; distance > 0; distance >>= 1) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xFFFFFFFFU, maximum, distance));
  }
  const float scale = maximum * (1.0F / 448.0F);
  if (lane == 0) {
    scales[static_cast<uint64_t>(row) * blocks + block] = scale;
  }
#pragma unroll
  for (int32_t part = 0; part < 4; ++part) {
    const uint8_t code = QuantizeCode(__fdiv_rn(values[part], scale));
    if constexpr (kIsDecode != 0) {
      reinterpret_cast<int32_t *>(activation)[first + part * 32 + lane] =
          (code & 127U) == 127U ? INT32_MAX : CodeInteger(code);
    } else {
      activation[first + part * 32 + lane] = code;
    }
  }
}

template <int32_t kIsAligned> __device__ inline uint32_t LoadCodes(const uint8_t *const codes) {
  if constexpr (kIsAligned != 0) {
    return *reinterpret_cast<const uint32_t *>(codes);
  } else {
    return static_cast<uint32_t>(codes[0]) | (static_cast<uint32_t>(codes[1]) << 8U) |
           (static_cast<uint32_t>(codes[2]) << 16U) | (static_cast<uint32_t>(codes[3]) << 24U);
  }
}

template <typename T, int32_t kIsAligned>
__global__ void GemvImpl(const uint8_t *const activation, const float *const activation_scales,
                         const uint8_t *const weight, const float *const scales, const int32_t rows,
                         const int32_t inner, const int32_t columns,
                         const int32_t block_reduction_groups, T *const output) {
  constexpr int32_t kWarpsPerColumn = 4;
  constexpr int32_t kLanesPerBlock = 8;
  constexpr int32_t kBlocksPerWarp = 32 / kLanesPerBlock;
  constexpr int32_t kValuesPerLane = 128 / kLanesPerBlock;
  constexpr int32_t kColumns = 4 / kWarpsPerColumn;
  constexpr int32_t kWindow = 256;
  __shared__ float partials[kColumns][kWindow];
  __shared__ int32_t weight_integers[256];
  for (int32_t code = threadIdx.x; code < 256; code += blockDim.x) {
    weight_integers[code] = CodeInteger(static_cast<uint8_t>(code));
  }
  __syncthreads();
  const int32_t lane = threadIdx.x % 32;
  const int32_t block_lane = lane % kLanesPerBlock;
  const int32_t warp = threadIdx.x / 32;
  const int32_t column = blockIdx.x * kColumns + warp / kWarpsPerColumn;
  const int32_t row = blockIdx.y;
  const int32_t blocks = inner / 128;
  float result = 0.0F;
  for (int32_t start = 0; start < blocks; start += kWindow) {
    const int32_t count = min(kWindow, blocks - start);
    for (int32_t block = warp % kWarpsPerColumn * kBlocksPerWarp + lane / kLanesPerBlock;
         block < count; block += kWarpsPerColumn * kBlocksPerWarp) {
      const int32_t logical = start + block;
      const int32_t physical =
          block_reduction_groups == 1
              ? logical
              : logical % block_reduction_groups * (blocks / block_reduction_groups) +
                    logical / block_reduction_groups;
      const int32_t first = physical * 128 + block_lane * kValuesPerLane;
      int64_t dot = 0;
      bool has_nan = false;
      const auto *const decoded = reinterpret_cast<const int4 *>(activation);
      const auto *const codes = weight + static_cast<uint64_t>(column) * inner + first;
      uint32_t weight_codes[kValuesPerLane / 4];
      if constexpr (kIsAligned != 0) {
        const uint4 packed = matmul_kernel::LoadGemvWeight<true>(codes);
        weight_codes[0] = packed.x;
        weight_codes[1] = packed.y;
        weight_codes[2] = packed.z;
        weight_codes[3] = packed.w;
      } else {
#pragma unroll
        for (int32_t vector = 0; vector < kValuesPerLane / 4; ++vector) {
          weight_codes[vector] = LoadCodes<kIsAligned>(codes + vector * 4);
        }
      }
#pragma unroll
      for (int32_t vector = 0; vector < kValuesPerLane / 4; ++vector) {
        const int4 packed = decoded[(static_cast<uint64_t>(row) * inner + first) / 4 + vector];
        const int32_t values[] = {packed.x, packed.y, packed.z, packed.w};
        const uint32_t packed_codes = weight_codes[vector];
#pragma unroll
        for (int32_t part = 0; part < 4; ++part) {
          const uint8_t encoded_weight = static_cast<uint8_t>(packed_codes >> (part * 8));
          has_nan = has_nan || values[part] == INT32_MAX || (encoded_weight & 127U) == 127U;
          dot += static_cast<int64_t>(values[part]) * weight_integers[encoded_weight];
        }
      }
      const uint32_t mask = __activemask();
#pragma unroll
      for (int32_t distance = kLanesPerBlock / 2; distance > 0; distance >>= 1) {
        dot += __shfl_down_sync(mask, dot, distance, kLanesPerBlock);
      }
      constexpr uint32_t kBlockMask = 0xFFFFFFFFU >> (32 - kLanesPerBlock);
      const bool is_nan = (__ballot_sync(mask, has_nan) &
                           (kBlockMask << (lane / kLanesPerBlock * kLanesPerBlock))) != 0U;
      const float partial = is_nan ? nanf("") : __ll2float_rn(dot) * (1.0F / 262144.0F);
      if (block_lane == 0) {
        const float activation_scale =
            activation_scales[static_cast<uint64_t>(row) * blocks + physical];
        const float weight_scale = scales[static_cast<uint64_t>(column / 128) * blocks + physical];
        partials[warp / kWarpsPerColumn][block] =
            __fmul_rn(__fmul_rn(partial, activation_scale), weight_scale);
      }
    }
    __syncthreads();
    if (threadIdx.x < kColumns) {
      for (int32_t block = 0; block < count; ++block) {
        result = __fadd_rn(result, partials[threadIdx.x][block]);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < kColumns && row < rows) {
    output[static_cast<uint64_t>(row) * columns + blockIdx.x * kColumns + threadIdx.x] =
        FromFloat<T>(result);
  }
}

__device__ inline void IntegerMma(const uint32_t *const a, const uint32_t *const b,
                                  int32_t *const c) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ inline void DecodeDigits(const uint32_t codes, const uint32_t *const lookup,
                                    uint32_t *const digits) {
  const uint32_t first = lookup[codes & 255U];
  const uint32_t second = lookup[(codes >> 8U) & 255U];
  const uint32_t third = lookup[(codes >> 16U) & 255U];
  const uint32_t fourth = lookup[codes >> 24U];
#pragma unroll
  for (int32_t digit = 0; digit < 3; ++digit) {
    const uint32_t selector = static_cast<uint32_t>(digit | ((digit + 4) << 4));
    digits[digit] = __byte_perm(__byte_perm(first, second, selector),
                                __byte_perm(third, fourth, selector), 0x5410U);
  }
}

template <typename T, int32_t kIsAligned, int32_t kRows>
__global__ void MatmulImpl(const uint8_t *const activation, const float *const activation_scales,
                           const uint8_t *const weight, const float *const scales,
                           const int32_t rows, const int32_t inner, const int32_t columns,
                           const int32_t block_reduction_groups, T *const output) {
  constexpr int32_t kTiles = kRows / 16;
  __shared__ __align__(16) int8_t left[3][kRows][144];
  __shared__ __align__(16) int8_t right[3][32][144];
  __shared__ bool activation_nan[kRows];
  __shared__ bool weight_nan[32];
  __shared__ uint32_t digit_lookup[256];
  const int32_t id = threadIdx.x;
  const int32_t lane = id % 32;
  const int32_t warp = id / 32;
  // Schedule row tiles together to reuse weights while preserving the launch grid limits.
  const uint64_t tile_id = static_cast<uint64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  const int32_t row_start = static_cast<int32_t>(tile_id % gridDim.y) * kRows;
  const int32_t column_start = static_cast<int32_t>(tile_id / gridDim.y) * 32;
  const int32_t warp_row = warp % kTiles;
  const int32_t warp_column = warp / kTiles * kTiles;
  float result[kTiles][4] = {};
  const int32_t blocks = inner / 128;
  for (int32_t code = id; code < 256; code += 128) {
    const int32_t integer = CodeInteger(static_cast<uint8_t>(code));
    uint32_t packed = 0U;
#pragma unroll
    for (int32_t digit = 0; digit < 3; ++digit) {
      packed |= static_cast<uint32_t>(static_cast<uint8_t>(IntegerDigit(integer, digit)))
                << (digit * 8);
    }
    digit_lookup[code] = packed;
  }
  __syncthreads();
  for (int32_t block = 0; block < blocks; ++block) {
    const int32_t physical =
        block_reduction_groups == 1
            ? block
            : block % block_reduction_groups * (blocks / block_reduction_groups) +
                  block / block_reduction_groups;
    const int32_t first = physical * 128;
    for (int32_t local_row = warp; local_row < 32; local_row += 4) {
      const uint32_t activation_codes =
          local_row < kRows && row_start + local_row < rows
              ? LoadCodes<1>(activation + static_cast<uint64_t>(row_start + local_row) * inner +
                             first + lane * 4)
              : 0U;
      const uint32_t weight_codes =
          local_row < 32 && column_start + local_row < columns
              ? LoadCodes<kIsAligned>(weight +
                                      static_cast<uint64_t>(column_start + local_row) * inner +
                                      first + lane * 4)
              : 0U;
      uint32_t left_digits[3];
      uint32_t right_digits[3];
      DecodeDigits(activation_codes, digit_lookup, left_digits);
      DecodeDigits(weight_codes, digit_lookup, right_digits);
#pragma unroll
      for (int32_t digit = 0; digit < 3; ++digit) {
        if (local_row < kRows) {
          reinterpret_cast<uint32_t *>(left[digit][local_row])[lane] = left_digits[digit];
        }
        if (local_row < 32) {
          reinterpret_cast<uint32_t *>(right[digit][local_row])[lane] = right_digits[digit];
        }
      }
      const bool is_activation_nan =
          __any_sync(0xFFFFFFFFU, __vcmpeq4(activation_codes & 0x7f7f7f7fU, 0x7f7f7f7fU) != 0U);
      const bool is_weight_nan =
          __any_sync(0xFFFFFFFFU, __vcmpeq4(weight_codes & 0x7f7f7f7fU, 0x7f7f7f7fU) != 0U);
      if (lane == 0) {
        if (local_row < kRows) {
          activation_nan[local_row] = is_activation_nan;
        }
        if (local_row < 32) {
          weight_nan[local_row] = is_weight_nan;
        }
      }
    }
    __syncthreads();
    // Base-128 digits and pairwise sums fit INT8. Six products recover the
    // three diagonal and three cross terms without rounding the exact dot.
    int32_t products[6][kTiles][4] = {};
#pragma unroll
    for (int32_t d = 0; d < 128; d += 32) {
      uint32_t a[3][4];
#pragma unroll
      for (int32_t digit = 0; digit < 3; ++digit) {
        LoadLeft(reinterpret_cast<const __half *>(&left[digit][0][0]), 72, warp_row, lane, d / 2,
                 a[digit]);
      }
#pragma unroll
      for (int32_t tile = 0; tile < kTiles; ++tile) {
        uint32_t b[3][2];
#pragma unroll
        for (int32_t digit = 0; digit < 3; ++digit) {
          LoadRight(reinterpret_cast<const __half *>(&right[digit][0][0]), 72, warp_column + tile,
                    lane, d / 2, b[digit]);
          IntegerMma(a[digit], b[digit], products[digit][tile]);
        }
#pragma unroll
        for (int32_t first_digit = 0; first_digit < 3; ++first_digit) {
#pragma unroll
          for (int32_t second_digit = first_digit + 1; second_digit < 3; ++second_digit) {
            uint32_t a_sum[4];
            uint32_t b_sum[2];
#pragma unroll
            for (int32_t item = 0; item < 4; ++item) {
              a_sum[item] = __vadd4(a[first_digit][item], a[second_digit][item]);
            }
#pragma unroll
            for (int32_t item = 0; item < 2; ++item) {
              b_sum[item] = __vadd4(b[first_digit][item], b[second_digit][item]);
            }
            IntegerMma(a_sum, b_sum, products[first_digit + second_digit + 2][tile]);
          }
        }
      }
    }
    const float weight_scale =
        scales[static_cast<uint64_t>(column_start / 128) * (inner / 128) + first / 128];
#pragma unroll
    for (int32_t tile = 0; tile < kTiles; ++tile) {
#pragma unroll
      for (int32_t item = 0; item < 4; ++item) {
        const int32_t local_row = warp_row * 16 + lane / 4 + item / 2 * 8;
        const int32_t local_column = (warp_column + tile) * 8 + lane % 4 * 2 + item % 2;
        const int32_t cross_low =
            products[3][tile][item] - products[0][tile][item] - products[1][tile][item];
        const int32_t cross_middle =
            products[4][tile][item] - products[0][tile][item] - products[2][tile][item];
        const int32_t cross_high =
            products[5][tile][item] - products[1][tile][item] - products[2][tile][item];
        const int32_t low = products[0][tile][item] + cross_low * 128;
        const int32_t high = products[1][tile][item] + cross_middle + cross_high * 128 +
                             products[2][tile][item] * 16384;
        const int64_t dot = static_cast<int64_t>(high) * 16384 + low;
        const float partial = activation_nan[local_row] || weight_nan[local_column]
                                  ? nanf("")
                                  : __ll2float_rn(dot) * (1.0F / 262144.0F);
        const float activation_scale =
            row_start + local_row < rows
                ? activation_scales[static_cast<uint64_t>(row_start + local_row) * blocks +
                                    first / 128]
                : 0.0F;
        result[tile][item] = __fadd_rn(
            result[tile][item], __fmul_rn(__fmul_rn(partial, activation_scale), weight_scale));
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int32_t tile = 0; tile < kTiles; ++tile) {
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t row = row_start + warp_row * 16 + lane / 4 + item / 2 * 8;
      const int32_t column = column_start + (warp_column + tile) * 8 + lane % 4 * 2 + item % 2;
      if (row < rows && column < columns) {
        output[static_cast<uint64_t>(row) * columns + column] = FromFloat<T>(result[tile][item]);
      }
    }
  }
}

} // namespace omni_runtime::kernels::fp8_kernel
