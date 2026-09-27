#pragma once

#include <kernels/common/kernel_type_utils.h>
#include <kernels/matmul_kernel/tensor_core.cuh>

namespace omni_runtime::kernels::quantized_kernel {

__device__ inline float Fp8Scale(const uint32_t code) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  if (code < 127U) {
    uint32_t packed;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packed) : "h"(static_cast<uint16_t>(code)));
    return __half2float(__ushort_as_half(static_cast<uint16_t>(packed))) * 0.5F;
  }
#endif
  const uint32_t exponent = (code >> 3U) & 15U;
  const uint32_t mantissa = code & 7U;
  return code == 0U || code == 127U ? 0.0F
         : exponent == 0U           ? static_cast<float>(mantissa) * 0.0009765625F
                          : __uint_as_float(((exponent + 119U) << 23U) | (mantissa << 20U));
}

__device__ inline int32_t Fp4Value(const uint32_t code) {
  const int32_t magnitude = (0xC8643210U >> ((code & 7U) * 4U)) & 15U;
  return code & 8U ? -magnitude : magnitude;
}

__device__ inline float ReadWeight(const uint8_t *const data, const int32_t type,
                                   const uint64_t index) {
  if (type == 0) {
    return reinterpret_cast<const float *>(data)[index];
  }
  if (type == 1) {
    return __half2float(reinterpret_cast<const __half *>(data)[index]);
  }
  if (type == 30) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(data)[index]);
  }
  if (type == 8) {
    const uint8_t *const block = data + index / 32U * 34U;
    return __half2float(*reinterpret_cast<const __half *>(block)) *
           static_cast<float>(reinterpret_cast<const int8_t *>(block + 2)[index % 32U]);
  }
  const uint8_t *const block = data + index / 64U * 36U;
  const uint32_t local = index % 64U;
  const uint32_t sub = local / 16U;
  const uint32_t element = local % 16U;
  const float factor = Fp8Scale(block[sub]);
  const uint32_t packed = block[4U + sub * 8U + element % 8U];
  const uint32_t code = (element < 8U ? packed : packed >> 4U) & 15U;
  return static_cast<float>(Fp4Value(code)) * factor;
}

template <typename T>
__global__ void PrepareBfloatImpl(const T *const input, const uint8_t *const weight,
                                  const uint64_t input_count, const uint64_t weight_count,
                                  __nv_bfloat16 *const packed_input,
                                  __nv_bfloat16 *const packed_weight) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < input_count) {
    packed_input[index] = __float2bfloat16_rn(ToFloat(input[index]));
  }
  if (index < weight_count) {
    packed_weight[index] = __float2bfloat16_rn(ReadWeight(weight, 40, index));
  }
}

template <typename T>
__global__ void ScaleRowsImpl(const float *const scales, const int32_t scale_stride,
                              const float weight_scale, const int32_t columns, const uint64_t count,
                              T *const output) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const float row_scale = scales == nullptr ? 1.0F : scales[index / columns * scale_stride];
    output[index] = FromFloat<T>((ToFloat(output[index]) * row_scale) * weight_scale);
  }
}

__device__ inline uint8_t NearestFp4Bits(const float value) {
  const float magnitude = fabsf(value);
  constexpr float kValues[] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  int32_t best = 0;
  float distance = magnitude;
#pragma unroll
  for (int32_t code = 1; code < 8; ++code) {
    const float error = fabsf(magnitude - kValues[code]);
    if (error < distance || (error == distance && code % 2 == 0)) {
      distance = error;
      best = code;
    }
  }
  return static_cast<uint8_t>(static_cast<uint32_t>(best) | (signbit(value) ? 8U : 0U));
}

__device__ inline int32_t NearestFp4(const float value) {
  return Fp4Value(NearestFp4Bits(value));
}

__device__ inline int32_t NearestFp8(const float value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  const float clamped = fminf(fmaxf(value, 0.0F), 448.0F);
  uint16_t packed;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(packed) : "f"(clamped), "f"(0.0F));
  return packed & 255U;
#else
  // E4M3's eight mantissa bins permit a bounded search around the FP32 exponent.
  const uint32_t bits = __float_as_uint(fminf(fmaxf(value, 0.0F), 448.0F));
  const int32_t exponent = static_cast<int32_t>((bits >> 23U) & 255U) - 127;
  const int32_t center =
      max(0, min(126, (exponent + 7) * 8 + static_cast<int32_t>((bits >> 20U) & 7U)));
  int32_t best = 0;
  float distance = value;
  for (int32_t code = max(0, center - 8); code <= min(126, center + 8); ++code) {
    const float error = fabsf(value - 2.0F * Fp8Scale(code));
    if (error < distance || (error == distance && code % 2 == 0)) {
      distance = error;
      best = code;
    }
  }
  return best;
#endif
}

template <typename T>
__global__ void QuantizeImpl(const T *const input, const int32_t weight_type, const int32_t rows,
                             const int32_t columns, const bool is_decode,
                             const bool is_ggml_rounding, const bool is_packed_nvfp4,
                             const bool is_packed_q8, T *const output, float *const scales) {
  const int32_t row = blockIdx.x;
  const uint64_t offset = static_cast<uint64_t>(row) * columns;
  if (weight_type == 8 || is_decode) {
    const int32_t lane = threadIdx.x % 32;
    for (int32_t first = blockIdx.y * blockDim.x + threadIdx.x / 32 * 32; first < columns;
         first += blockDim.x * gridDim.y) {
      const float value = ToFloat(input[offset + first + lane]);
      float maximum = fabsf(value);
#pragma unroll
      for (int32_t distance = 16; distance > 0; distance >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xFFFFFFFFU, maximum, distance));
      }
      const float inverse =
          maximum > 0.0F ? (is_ggml_rounding ? __fdividef(127.0F, maximum) : 127.0F / maximum)
                         : 0.0F;
      const float scale = !is_ggml_rounding ? maximum / 127.0F
                          : is_decode       ? __fdividef(maximum, 127.0F)
                                            : (maximum > 0.0F ? __fdividef(1.0F, inverse) : 0.0F);
      const float code =
          is_decode ? (maximum > 0.0F
                           ? roundf(is_ggml_rounding ? __fdividef(value, scale) : value / scale)
                           : 0.0F)
                    : roundf(value * inverse);
      if (is_packed_q8) {
        reinterpret_cast<int8_t *>(output)[offset + first + lane] = static_cast<int8_t>(code);
      } else {
        output[offset + first + lane] = FromFloat<T>(code);
      }
      if (lane == 0) {
        scales[static_cast<uint64_t>(row) * (columns / 32) + first / 32] =
            is_decode ? __half2float(__float2half_rn(scale)) : scale;
      }
    }
    return;
  }
  float maximum = 0.0F;
  for (int32_t d = threadIdx.x; d < columns; d += blockDim.x) {
    maximum = fmaxf(maximum, fabsf(ToFloat(input[offset + d])));
  }
  __shared__ float scratch[8];
  const float row_maximum = BlockReduceMax(maximum, scratch);
  const float row_scale = is_ggml_rounding ? row_maximum * (1.0F / 2688.0F) : row_maximum / 2688.0F;
  const float inverse_row =
      row_scale > 0.0F ? (is_ggml_rounding ? __fdividef(1.0F, row_scale) : 1.0F / row_scale) : 0.0F;
  for (int32_t first = threadIdx.x * 16; first < columns; first += blockDim.x * 16) {
    float values[16];
    float local_maximum = 0.0F;
#pragma unroll
    for (int32_t j = 0; j < 16; ++j) {
      values[j] = ToFloat(input[offset + first + j]) * inverse_row;
      local_maximum = fmaxf(local_maximum, fabsf(values[j]));
    }
    const int32_t center =
        NearestFp8(is_ggml_rounding ? local_maximum * (1.0F / 6.0F) : local_maximum / 6.0F);
    constexpr int32_t kOffsets[] = {0, -1, 1, -2, 2};
    float best_scale = 0.0F;
    uint8_t best_scale_code = 0U;
    float best_error = __int_as_float(0x7F800000);
#pragma unroll
    for (int32_t candidate = 0; candidate < 5; ++candidate) {
      const int32_t code = center + kOffsets[candidate];
      if (code < 0 || code > 126) {
        continue;
      }
      const float scale = Fp8Scale(code);
      const float inverse =
          scale > 0.0F ? (is_ggml_rounding ? __fdividef(0.5F, scale) : 0.5F / scale) : 0.0F;
      float error = 0.0F;
#pragma unroll
      for (int32_t j = 0; j < 16; ++j) {
        const float difference =
            fabsf(values[j]) - fabsf(static_cast<float>(NearestFp4(values[j] * inverse))) * scale;
        error = fmaf(difference, difference, error);
      }
      if (error < best_error) {
        best_error = error;
        best_scale = scale;
        best_scale_code = static_cast<uint8_t>(code);
      }
    }
    const float inverse =
        best_scale > 0.0F
            ? inverse_row * (is_ggml_rounding ? __fdividef(0.5F, best_scale) : 0.5F / best_scale)
            : 0.0F;
    if (is_packed_nvfp4) {
      uint8_t *const block = reinterpret_cast<uint8_t *>(output) +
                             (static_cast<uint64_t>(row) * (columns / 64) + first / 64) * 36U;
      const int32_t sub = first % 64 / 16;
      block[sub] = best_scale_code;
#pragma unroll
      for (int32_t j = 0; j < 8; ++j) {
        const uint8_t low = NearestFp4Bits(ToFloat(input[offset + first + j]) * inverse);
        const uint8_t high = NearestFp4Bits(ToFloat(input[offset + first + j + 8]) * inverse);
        block[4 + sub * 8 + j] = low | (high << 4U);
      }
    } else {
#pragma unroll
      for (int32_t j = 0; j < 16; ++j) {
        output[offset + first + j] = FromFloat<T>(
            static_cast<float>(NearestFp4(ToFloat(input[offset + first + j]) * inverse)) *
            best_scale);
      }
    }
    scales[static_cast<uint64_t>(row) * (columns / 16) + first / 16] = row_scale;
  }
}

template <int32_t kType>
__device__ inline int32_t WeightCode(const uint8_t *const weight, const uint64_t index) {
  if constexpr (kType == 8) {
    return reinterpret_cast<const int8_t *>(weight + index / 32U * 34U + 2U)[index % 32U];
  } else {
    const uint8_t *const block = weight + index / 64U * 36U;
    const uint32_t local = index % 64U;
    const uint32_t packed = block[4U + local / 16U * 8U + local % 8U];
    return Fp4Value((packed >> (local % 16U < 8U ? 0U : 4U)) & 15U);
  }
}

template <int32_t kType>
__device__ inline float WeightScale(const uint8_t *const weight, const uint64_t index) {
  if constexpr (kType == 8) {
    return __half2float(*reinterpret_cast<const __half *>(weight + index / 32U * 34U));
  } else {
    return Fp8Scale(weight[index / 64U * 36U + index % 64U / 16U]);
  }
}

__device__ inline uint32_t LoadPackedBytes(const uint8_t *const data) {
  if (reinterpret_cast<uintptr_t>(data) % 4U == 0U) {
    return *reinterpret_cast<const uint32_t *>(data);
  }
  if (reinterpret_cast<uintptr_t>(data) % 2U == 0U) {
    return static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(data)) |
           (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(data + 2)) << 16U);
  }
  return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8U) |
         (static_cast<uint32_t>(data[2]) << 16U) | (static_cast<uint32_t>(data[3]) << 24U);
}

template <int32_t kType>
__device__ inline uint32_t PackedWeightCodes(const uint8_t *const weight, const uint64_t index) {
  if constexpr (kType == 8) {
    const auto *const words =
        reinterpret_cast<const uint16_t *>(weight + index / 32U * 34U + 2U + index % 32U);
    return static_cast<uint32_t>(words[0]) | (static_cast<uint32_t>(words[1]) << 16U);
  } else {
    const uint32_t local = index % 64U;
    const uint32_t packed =
        LoadPackedBytes(weight + index / 64U * 36U + 4U + local / 16U * 8U + local % 8U);
    const uint32_t shift = local % 16U < 8U ? 0U : 4U;
    uint32_t result = 0U;
#pragma unroll
    for (uint32_t item = 0U; item < 4U; ++item) {
      result |= static_cast<uint32_t>(
                    static_cast<uint8_t>(Fp4Value((packed >> (item * 8U + shift)) & 15U)))
                << (item * 8U);
    }
    return result;
  }
}

__device__ inline uint2 Fp4Codes2(const uint32_t packed) {
  uint32_t values[2];
  const uint32_t selector = 0x32103210U | ((packed & 0x88888888U) >> 1U);
#pragma unroll
  for (uint32_t part = 0U; part < 2U; ++part) {
    const uint32_t codes = packed >> (part * 16U);
    const uint32_t low = __byte_perm(0x03020100U, 0x0C080604U, codes & 0x7777U);
    const uint32_t high = __byte_perm(0xFDFEFF00U, 0xF4F8FAFCU, codes & 0x7777U);
    values[part] = __byte_perm(low, high, (selector >> (part * 16U)) & 0x7777U);
  }
  return make_uint2(__byte_perm(values[0], values[1], 0x6420U),
                    __byte_perm(values[0], values[1], 0x7531U));
}

template <typename T, int32_t kIsPacked>
__device__ inline uint32_t ActivationCodes(const T *const input, const int64_t first) {
  if constexpr (kIsPacked != 0) {
    return LoadPackedBytes(reinterpret_cast<const uint8_t *>(input) + first);
  }
  uint32_t packed = 0U;
#pragma unroll
  for (int32_t item = 0; item < 4; ++item) {
    packed |= static_cast<uint32_t>(
                  static_cast<uint8_t>(static_cast<int32_t>(ToFloat(input[first + item]))))
              << (item * 8U);
  }
  return packed;
}

__device__ inline int32_t SharedActivationWord(const int32_t word, const int32_t mask) {
  return word ^ ((word >> 5) & mask);
}

template <typename T, int32_t kType, int32_t kWarps, int32_t kIsPacked, int32_t kStageWeights>
__global__ void ScaledGemvImpl(const T *const input, const uint8_t *const global_weight,
                               const float *const activation_scales, const float global_scale,
                               const int32_t inner, const int32_t columns, T *const output) {
  constexpr int32_t kColumns = 8 / kWarps;
  const int32_t column = blockIdx.x * kColumns + threadIdx.x / (32 * kWarps);
  const int32_t warp = threadIdx.x / 32 % kWarps;
  const int32_t lane = threadIdx.x % 32;
  float sum = 0.0F;
  const int64_t row_offset = static_cast<int64_t>(blockIdx.y) * inner;
  const int64_t scale_offset = static_cast<int64_t>(blockIdx.y) * (inner / 32);
  // Permute complete 32-word banks to avoid the FP4 eight-way stride conflict.
  const int32_t swizzle_mask = inner % 128 == 0 ? 31 : 0;
  extern __shared__ __align__(16) uint32_t packed_input[];
  const int32_t weight_column = kStageWeights != 0 ? threadIdx.x / (32 * kWarps) : column;
  const uint8_t *const weight =
      kStageWeights != 0 ? reinterpret_cast<const uint8_t *>(packed_input) : global_weight;
  if constexpr (kStageWeights != 0) {
    static_assert(kType == 8 && kIsPacked != 0);
    const int32_t row_bytes = inner / 32 * 34;
    for (int32_t offset = threadIdx.x * 16; offset < kColumns * row_bytes;
         offset += blockDim.x * 16) {
      const uint32_t address = static_cast<uint32_t>(
          __cvta_generic_to_shared(reinterpret_cast<uint8_t *>(packed_input) + offset));
      const bool is_valid = blockIdx.x * kColumns + offset / row_bytes < columns;
      asm volatile(
          "cp.async.cg.shared.global [%0], [%1], 16, %2;"
          :
          : "r"(address),
            "l"(global_weight + static_cast<uint64_t>(blockIdx.x) * kColumns * row_bytes + offset),
            "r"(is_valid ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
  }
  if (kIsPacked == 0 && inner <= 32768) {
    for (int32_t d = threadIdx.x * 4; d < inner; d += blockDim.x * 4) {
      packed_input[SharedActivationWord(d / 4, swizzle_mask)] =
          ActivationCodes<T, 0>(input, row_offset + d);
    }
    __syncthreads();
  }
  constexpr int32_t kValues = kType == 8 ? 8 : 32;
  for (int32_t first = (warp * 32 + lane) * kValues; column < columns && first < inner;
       first += 32 * kWarps * kValues) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1210
    if constexpr (kType == 8 && kStageWeights == 0) {
      const int32_t future = first + 2 * 32 * kWarps * kValues;
      if (future < inner) {
        const uint8_t *const block =
            weight + (static_cast<uint64_t>(column) * inner + future) / 32U * 34U;
        asm volatile("prefetch.global.L2 [%0];" : : "l"(block));
      }
    }
#endif
    float partial = 0.0F;
#pragma unroll
    for (int32_t sub = 0; sub < kValues; sub += kType == 8 ? 8 : 16) {
      int32_t dot = 0;
      if constexpr (kType == 8) {
#pragma unroll
        for (int32_t group = 0; group < 8; group += 4) {
          const int32_t d = first + group;
          const uint32_t w =
              PackedWeightCodes<kType>(weight, static_cast<uint64_t>(weight_column) * inner + d);
          const uint32_t x = kIsPacked == 0 && inner <= 32768
                                 ? packed_input[SharedActivationWord(d / 4, swizzle_mask)]
                                 : ActivationCodes<T, kIsPacked>(input, row_offset + d);
          dot = __dp4a(static_cast<int32_t>(x), static_cast<int32_t>(w), dot);
        }
      } else {
        const uint64_t index = static_cast<uint64_t>(weight_column) * inner + first + sub;
        const uint8_t *const codes = weight + index / 64U * 36U + 4U + index % 64U / 16U * 8U;
#pragma unroll
        for (int32_t group = 0; group < 2; ++group) {
          const uint2 w = Fp4Codes2(LoadPackedBytes(codes + group * 4));
          const int32_t d = first + sub + group * 4;
          const uint32_t low = kIsPacked == 0 && inner <= 32768
                                   ? packed_input[SharedActivationWord(d / 4, swizzle_mask)]
                                   : ActivationCodes<T, kIsPacked>(input, row_offset + d);
          const uint32_t high = kIsPacked == 0 && inner <= 32768
                                    ? packed_input[SharedActivationWord(d / 4 + 2, swizzle_mask)]
                                    : ActivationCodes<T, kIsPacked>(input, row_offset + d + 8);
          dot = __dp4a(static_cast<int32_t>(low), static_cast<int32_t>(w.x), dot);
          dot = __dp4a(static_cast<int32_t>(high), static_cast<int32_t>(w.y), dot);
        }
      }
      const float scale =
          WeightScale<kType>(weight, static_cast<uint64_t>(weight_column) * inner + first + sub) *
          activation_scales[scale_offset + (first + sub) / 32];
      if constexpr (kType == 8) {
        sum = fmaf(static_cast<float>(dot), scale, sum);
      } else {
        partial += static_cast<float>(dot) * scale;
      }
    }
    if constexpr (kType != 8) {
      sum += partial;
    }
  }
  if constexpr (kWarps > 1) {
    __shared__ float partials[8][32];
    partials[threadIdx.x / 32][lane] = sum;
    __syncthreads();
    if (warp != 0) {
      return;
    }
#pragma unroll
    for (int32_t part = 1; part < kWarps; ++part) {
      sum += partials[threadIdx.x / 32 + part][lane];
    }
  }
  sum = WarpReduceSum(sum);
  if (lane == 0 && column < columns) {
    output[static_cast<int64_t>(blockIdx.y) * columns + column] = FromFloat<T>(sum * global_scale);
  }
}

template <typename T, typename M>
__device__ void Matmul(const T *const input, const uint8_t *const weight, const int32_t weight_type,
                       const float weight_scale, const int32_t rows, const int32_t inner,
                       const int32_t columns, const float *const row_scales,
                       const int32_t scale_stride, T *const output);

template <typename M>
__device__ inline void LoadLeft(const M *const tile, const int32_t stride, const int32_t warp,
                                const int32_t lane, const int32_t first, uint32_t (&values)[4]) {
  const uint32_t address = static_cast<uint32_t>(
      __cvta_generic_to_shared(tile + (warp * 16 + lane % 16) * stride + first + lane / 16 * 8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
               : "r"(address));
}

template <typename M>
__device__ inline void LoadRight(const M *const tile, const int32_t stride, const int32_t column,
                                 const int32_t lane, const int32_t first, uint32_t (&values)[2]) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(
      tile + (column * 8 + lane % 8) * stride + first + (lane / 8 % 2) * 8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(values[0]), "=r"(values[1])
               : "r"(address));
}

template <typename T>
__global__ void PackedNvfp4GemvImpl(const uint8_t *const input, const uint8_t *const weight,
                                    const float *const scales, const float global_scale,
                                    const int32_t rows, const int32_t inner, const int32_t columns,
                                    T *const output) {
  const int32_t column = blockIdx.x * 4 + threadIdx.x / 32;
  const int32_t row = blockIdx.y;
  float sum = 0.0F;
  if (column < columns) {
    for (int32_t d = threadIdx.x % 32; d < inner; d += 32) {
      sum = fmaf(ReadWeight(input, 40, static_cast<uint64_t>(row) * inner + d),
                 ReadWeight(weight, 40, static_cast<uint64_t>(column) * inner + d), sum);
    }
  }
  sum = WarpReduceSum(sum);
  if (threadIdx.x % 32 == 0 && column < columns) {
    output[static_cast<uint64_t>(row) * columns + column] =
        FromFloat<T>((sum * scales[static_cast<uint64_t>(row) * (inner / 16)]) * global_scale);
  }
}

__device__ inline void Nvfp4Mma(const uint32_t (&a)[4], const uint32_t (&b)[2],
                                const uint32_t a_scale, const uint32_t b_scale,
                                float (&output)[4]) {
#if __CUDA_ARCH__ >= 1200
  float partial[4] = {};
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
               "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
               "%10, {0,0}, %11, {0,0};"
               : "+f"(partial[0]), "+f"(partial[1]), "+f"(partial[2]), "+f"(partial[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(a_scale),
                 "r"(b_scale));
  // Match the reference build's K64 truncation followed by FP32 addition.
#pragma unroll
  for (int32_t item = 0; item < 4; ++item) {
    output[item] += partial[item];
  }
#else
  asm volatile("trap;");
#endif
}

template <int32_t kIsAsync>
__device__ inline void StageWord(const uint8_t *const source, const bool is_valid,
                                 uint32_t *const destination) {
  if constexpr (kIsAsync != 0) {
    const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
                 :
                 : "r"(address), "l"(source), "r"(is_valid ? 4 : 0));
  } else {
    *destination = is_valid ? LoadPackedBytes(source) : 0U;
  }
}

// Mode 2 uses FP32 output scratch when a tile intersects at most two partitions.
template <typename T, int32_t kIsSplitK, int32_t kColumns, int32_t kRows, int32_t kIsAsync>
__global__ __launch_bounds__(kRows * 2, kRows == 16 ? 4 : 2) void PackedNvfp4MatmulImpl(
    const uint8_t *const input, const uint8_t *const weight, const float *const scales,
    const float global_scale, const int32_t rows, const int32_t inner, const int32_t columns,
    const int32_t token_tile, const int32_t stream_groups, T *const output) {
  constexpr int32_t kStageBlocks = kIsAsync != 0 ? 2 : kRows == 16 ? 1 : 4;
  constexpr int32_t kBuffers = kIsAsync != 0 ? 2 : 1;
  constexpr int32_t kStride = kStageBlocks * 16 + 8;
  __shared__ __align__(16) uint16_t left[kBuffers][kRows][kStride];
  __shared__ __align__(16) uint16_t right[kBuffers][kColumns][kStride];
  __shared__ uint32_t left_scales[kBuffers][kRows][kStageBlocks];
  __shared__ uint32_t right_scales[kBuffers][kColumns][kStageBlocks];
  const int32_t id = threadIdx.x;
  const int32_t lane = id % 32;
  const int32_t warp = id / 32;
  const int32_t tile_rows = kIsSplitK != 0 ? token_tile : kRows;
  const int32_t row_start = blockIdx.x * tile_rows;
  const int32_t column_start = blockIdx.y * kColumns;
  const int32_t blocks = inner / 64;
  float accumulators[kColumns / 8][4] = {};
  float last[kColumns / 8][4] = {};
  float previous[kColumns / 8][4] = {};
  bool has_last = false;
  const int32_t parts = kIsSplitK != 0 ? stream_groups : 1;
  int32_t first_part = 0;
  int32_t last_part = parts - 1;
  if constexpr (kIsSplitK != 0) {
    const int64_t token_tiles = (rows - 1) / token_tile + 1;
    const int64_t tile = static_cast<int64_t>(column_start / 128) * token_tiles + blockIdx.x;
    const int64_t total_blocks = ((columns - 1) / 128 + 1) * token_tiles * blocks;
    first_part = static_cast<int32_t>(tile * blocks * stream_groups / total_blocks);
    last_part = static_cast<int32_t>(
        min(static_cast<int64_t>(parts - 1), (tile + 1) * blocks * stream_groups / total_blocks));
  }
  for (int32_t part = last_part; part >= first_part; --part) {
    int32_t first_block = 0;
    int32_t stop_block = blocks;
    if constexpr (kIsSplitK != 0) {
      const int64_t token_tiles = (rows - 1) / token_tile + 1;
      const int64_t tile = static_cast<int64_t>(column_start / 128) * token_tiles + blockIdx.x;
      const int64_t total_blocks = ((columns - 1) / 128 + 1) * token_tiles * blocks;
      int64_t first = static_cast<int64_t>(part) * total_blocks / stream_groups;
      int64_t stop = static_cast<int64_t>(part + 1) * total_blocks / stream_groups;
      first -= first % blocks % 8;
      stop -= stop % blocks % 8;
      first_block = static_cast<int32_t>(max(first, tile * blocks) - tile * blocks);
      stop_block = static_cast<int32_t>(min(stop, (tile + 1) * blocks) - tile * blocks);
      if (first_block >= stop_block) {
        continue;
      }
#pragma unroll
      for (int32_t tile = 0; tile < kColumns / 8; ++tile) {
#pragma unroll
        for (int32_t item = 0; item < 4; ++item) {
          accumulators[tile][item] = 0.0F;
        }
      }
    }
    const auto stage_input = [&](const int32_t first, const int32_t stage) {
      constexpr int32_t kLoadRows = kRows > kColumns ? kRows : kColumns;
      for (int32_t index = id; index < kLoadRows * kStageBlocks * 8; index += kRows * 2) {
        const int32_t r = index / (kStageBlocks * 8);
        const int32_t local = index % (kStageBlocks * 8);
        const int32_t part = local / 8;
        const int32_t word = index % 8;
        const uint8_t *const left_block =
            input + (static_cast<uint64_t>(row_start + r) * blocks + first + part) * 36U;
        const uint8_t *const right_block =
            weight + (static_cast<uint64_t>(column_start + r) * blocks + first + part) * 36U;
        if (r < kRows) {
          StageWord<kIsAsync>(left_block + 4 + word * 4,
                              r < tile_rows && row_start + r < rows && first + part < stop_block,
                              &reinterpret_cast<uint32_t *>(left[stage][r])[local]);
        }
        if (r < kColumns) {
          StageWord<kIsAsync>(right_block + 4 + word * 4,
                              column_start + r < columns && first + part < stop_block,
                              &reinterpret_cast<uint32_t *>(right[stage][r])[local]);
        }
        if (word == 0) {
          if (r < kRows) {
            StageWord<kIsAsync>(left_block,
                                r < tile_rows && row_start + r < rows && first + part < stop_block,
                                &left_scales[stage][r][part]);
          }
          if (r < kColumns) {
            StageWord<kIsAsync>(right_block,
                                column_start + r < columns && first + part < stop_block,
                                &right_scales[stage][r][part]);
          }
        }
      }
    };
    if constexpr (kIsAsync != 0) {
      stage_input(first_block, 0);
      asm volatile("cp.async.commit_group;");
    }
    int32_t stage = 0;
    for (int32_t first = first_block; first < stop_block; first += kStageBlocks) {
      if constexpr (kIsAsync != 0) {
        const bool has_next = first + kStageBlocks < stop_block;
        if (has_next) {
          stage_input(first + kStageBlocks, stage ^ 1);
          asm volatile("cp.async.commit_group;");
          asm volatile("cp.async.wait_group 1;");
        } else {
          asm volatile("cp.async.wait_group 0;");
        }
      } else {
        stage_input(first, 0);
      }
      __syncthreads();
      for (int32_t part = 0; part < kStageBlocks && first + part < stop_block; ++part) {
        uint32_t a[4];
        LoadLeft(&left[stage][0][0], kStride, warp, lane, part * 16, a);
        const uint32_t a_scale = left_scales[stage][warp * 16 + lane / 4 + lane % 2 * 8][part];
#pragma unroll
        for (int32_t tile = 0; tile < kColumns / 8; ++tile) {
          uint32_t b[2];
          LoadRight(&right[stage][0][0], kStride, tile, lane, part * 16, b);
          Nvfp4Mma(a, b, a_scale, right_scales[stage][tile * 8 + lane / 4][part],
                   accumulators[tile]);
        }
      }
      __syncthreads();
      if constexpr (kIsAsync != 0) {
        stage ^= 1;
      }
    }
    if constexpr (kIsSplitK != 0) {
#pragma unroll
      for (int32_t tile = 0; tile < kColumns / 8; ++tile) {
#pragma unroll
        for (int32_t item = 0; item < 4; ++item) {
          const int32_t local_row = warp * 16 + lane / 4 + item / 2 * 8;
          const int32_t row = row_start + local_row;
          const float partial =
              local_row < tile_rows && row < rows
                  ? accumulators[tile][item] * scales[static_cast<uint64_t>(row) * (inner / 16)]
                  : 0.0F;
          if constexpr (kIsSplitK == 2) {
            const int32_t column = column_start + tile * 8 + lane % 4 * 2 + item % 2;
            if (local_row < tile_rows && row < rows && column < columns) {
              const uint64_t index = static_cast<uint64_t>(row) * columns + column;
              const float rounded = partial + 0.0F;
              output[index] = FromFloat<T>(has_last ? ToFloat(output[index]) + rounded : rounded);
            }
          } else if (!has_last) {
            last[tile][item] = partial;
          } else {
            previous[tile][item] += partial;
          }
        }
      }
      has_last = true;
    }
  }
#pragma unroll
  for (int32_t tile = 0; tile < kColumns / 8; ++tile) {
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t local_row = warp * 16 + lane / 4 + item / 2 * 8;
      const int32_t row = row_start + local_row;
      const int32_t column = column_start + tile * 8 + lane % 4 * 2 + item % 2;
      if (local_row < tile_rows && row < rows && column < columns) {
        const float sum =
            kIsSplitK == 2 ? ToFloat(output[static_cast<uint64_t>(row) * columns + column])
            : kIsSplitK != 0
                ? last[tile][item] + previous[tile][item]
                : accumulators[tile][item] * scales[static_cast<uint64_t>(row) * (inner / 16)];
        output[static_cast<uint64_t>(row) * columns + column] = FromFloat<T>(sum * global_scale);
      }
    }
  }
}
template <typename T, int32_t kType>
__global__ void ScaledMatmulImpl(const T *const input, const uint8_t *const weight,
                                 const float *const activation_scales,
                                 const int32_t activation_block_size, const float global_scale,
                                 const int32_t rows, const int32_t inner, const int32_t columns,
                                 T *const output) {
  if constexpr (kType == 40) {
    if (activation_block_size == 16) {
      Matmul<T, __nv_bfloat16>(input, weight, kType, global_scale, rows, inner, columns,
                               activation_scales, inner / 16, output);
      return;
    }
  }
  constexpr int32_t kBlock = kType == 8 ? 32 : 16;
  __shared__ __align__(16) __half left[64][40];
  __shared__ __align__(16) __half right[64][40];
  const int32_t id = threadIdx.x;
  const int32_t lane = id % 32;
  const int32_t warp = id / 32;
  const int32_t row_start = blockIdx.y * 64;
  const int32_t column_start = blockIdx.x * 64;
  float accumulators[8][4] = {};
  for (int32_t first = 0; first < inner; first += kBlock) {
    for (int32_t index = id; index < 64 * 32; index += 128) {
      const int32_t r = index / 32;
      const int32_t local = index % 32;
      const int32_t d = first + local;
      left[r][local] =
          __float2half_rn(local < kBlock && row_start + r < rows
                              ? ToFloat(input[static_cast<uint64_t>(row_start + r) * inner + d])
                              : 0.0F);
      right[r][local] =
          __float2half_rn(local < kBlock && column_start + r < columns
                              ? static_cast<float>(WeightCode<kType>(
                                    weight, static_cast<uint64_t>(column_start + r) * inner + d))
                              : 0.0F);
    }
    __syncthreads();
    float partial[8][4] = {};
#pragma unroll
    for (int32_t d = 0; d < kBlock; d += 16) {
      uint32_t a[4];
      LoadLeft(&left[0][0], 40, warp, lane, d, a);
#pragma unroll
      for (int32_t tile = 0; tile < 8; ++tile) {
        uint32_t b[2];
        LoadRight(&right[0][0], 40, tile, lane, d, b);
        TensorMma<__half>(a, b, partial[tile]);
      }
    }
#pragma unroll
    for (int32_t tile = 0; tile < 8; ++tile) {
#pragma unroll
      for (int32_t item = 0; item < 4; ++item) {
        const int32_t row = row_start + warp * 16 + lane / 4 + item / 2 * 8;
        const int32_t column = column_start + tile * 8 + lane % 4 * 2 + item % 2;
        if (row < rows && column < columns) {
          const float scale =
              WeightScale<kType>(weight, static_cast<uint64_t>(column) * inner + first) *
              activation_scales[static_cast<uint64_t>(row) * (inner / activation_block_size) +
                                first / activation_block_size];
          accumulators[tile][item] = fmaf(partial[tile][item], scale, accumulators[tile][item]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int32_t tile = 0; tile < 8; ++tile) {
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t row = row_start + warp * 16 + lane / 4 + item / 2 * 8;
      const int32_t column = column_start + tile * 8 + lane % 4 * 2 + item % 2;
      if (row < rows && column < columns) {
        output[static_cast<uint64_t>(row) * columns + column] =
            FromFloat<T>(accumulators[tile][item] * global_scale);
      }
    }
  }
}

template <typename T, int32_t kIsSplitK, int32_t kIsPacked, int32_t kColumns, int32_t kIsAsync>
__global__
__launch_bounds__(kIsSplitK == 2 && kColumns == 64 ? 128 : 256,
                  kIsSplitK == 2 && kColumns == 64 ? 3
                  : kIsSplitK == 0
                      ? 2
                      : 1) void Q8MatmulImpl(const T *const input, const uint8_t *const weight,
                                             const float *const scales, const float global_scale,
                                             const int32_t rows, const int32_t inner,
                                             const int32_t columns, const int32_t token_tile,
                                             const int32_t stream_groups, T *const output) {
  constexpr int32_t kRows = kIsSplitK == 1 ? 8 : kIsSplitK == 2 ? 64 : 128;
  constexpr int32_t kSharedRows = kIsSplitK == 1 ? 16 : kRows;
  constexpr int32_t kTiles = kIsSplitK == 1 ? 2 : kIsSplitK == 2 ? 8 : 16;
  constexpr int32_t kStageBlocks = kIsAsync != 0 ? 2 : 4;
  constexpr int32_t kBuffers = kIsAsync != 0 ? 2 : 1;
  constexpr int32_t kStride = kStageBlocks * 8 + 4;
  constexpr int32_t kThreads = kIsSplitK == 2 && kColumns == 64 ? 128 : 256;
  constexpr int32_t kLoadRows = kSharedRows > kColumns ? kSharedRows : kColumns;
  __shared__ __align__(16) uint32_t left[kBuffers][kSharedRows][kStride];
  __shared__ __align__(16) uint32_t right[kIsAsync != 0 ? 1 : kColumns][kStride];
  __shared__ float input_scales[kBuffers][kSharedRows][kStageBlocks];
  __shared__ float weight_scales[kIsAsync != 0 ? 1 : kColumns][kStageBlocks];
  __shared__ __align__(16) uint8_t raw_right[kBuffers][kIsAsync != 0 ? kColumns : 1]
                                            [kIsAsync != 0 ? kStageBlocks * 34 : 1];
  const int32_t id = threadIdx.x;
  const int32_t lane = id % 32;
  const int32_t warp = id / 32;
  const int32_t warp_row = kIsSplitK == 1 ? 0 : (warp % (kIsSplitK == 2 ? 4 : 8)) * 16;
  const int32_t warp_column = kIsSplitK == 1 ? warp * 16 : kIsSplitK == 2 ? warp / 4 * 64 : 0;
  // Adjacent CTAs reuse the same quantized weights across token tiles.
  const int32_t row_start = blockIdx.x * kRows;
  const int32_t column_start = blockIdx.y * kColumns;
  float accumulators[kTiles][4] = {};
  float last[kTiles][4] = {};
  float previous[kTiles][4] = {};
  bool has_last = false;
  const int32_t parts = kIsSplitK != 0 ? stream_groups : 1;
  int32_t first_part = 0;
  int32_t last_part = parts - 1;
  if constexpr (kIsSplitK != 0) {
    const int64_t blocks = inner / 32;
    const int64_t token_tiles = (rows - 1) / token_tile + 1;
    const int64_t tile = column_start / 128 * token_tiles + row_start / token_tile;
    const int64_t total_blocks = ((columns - 1) / 128 + 1) * token_tiles * blocks;
    first_part = static_cast<int32_t>(tile * blocks * stream_groups / total_blocks);
    last_part = static_cast<int32_t>(
        min(static_cast<int64_t>(parts - 1), (tile + 1) * blocks * stream_groups / total_blocks));
  }
  for (int32_t part = last_part; part >= first_part; --part) {
    int32_t first_inner = 0;
    int32_t stop_inner = inner;
    if constexpr (kIsSplitK != 0) {
      const int64_t blocks = inner / 32;
      const int64_t token_tiles = (rows - 1) / token_tile + 1;
      const int64_t tile = column_start / 128 * token_tiles + row_start / token_tile;
      const int64_t total_blocks = ((columns - 1) / 128 + 1) * token_tiles * blocks;
      int64_t first = static_cast<int64_t>(part) * total_blocks / stream_groups;
      int64_t stop = static_cast<int64_t>(part + 1) * total_blocks / stream_groups;
      first -= first % blocks % 8;
      stop -= stop % blocks % 8;
      first_inner = static_cast<int32_t>(max(first, tile * blocks) - tile * blocks) * 32;
      stop_inner = static_cast<int32_t>(min(stop, (tile + 1) * blocks) - tile * blocks) * 32;
    }
    if (first_inner >= stop_inner) {
      continue;
    }
#pragma unroll
    for (int32_t tile = 0; tile < kTiles; ++tile) {
#pragma unroll
      for (int32_t item = 0; item < 4; ++item) {
        accumulators[tile][item] = 0.0F;
      }
    }
    const auto stage_input = [&](const int32_t first, const int32_t stage) {
      if constexpr (kIsAsync != 0) {
        static_assert(kIsPacked != 0);
        for (int32_t index = id; index < kSharedRows * kStageBlocks * 8; index += kThreads) {
          const int32_t r = index / (kStageBlocks * 8);
          const int32_t local = index % (kStageBlocks * 8);
          const int32_t d = first + local * 4;
          StageWord<1>(reinterpret_cast<const uint8_t *>(input) +
                           static_cast<int64_t>(row_start + r) * inner + d,
                       row_start + r < rows && d < stop_inner, &left[stage][r][local]);
        }
        constexpr int32_t kRawWords = kStageBlocks * 34 / 4;
        for (int32_t index = id; index < kColumns * kRawWords; index += kThreads) {
          const int32_t r = index / kRawWords;
          const int32_t word = index % kRawWords;
          StageWord<1>(
              weight + (static_cast<uint64_t>(column_start + r) * inner + first) / 32U * 34U +
                  word * 4,
              column_start + r < columns, reinterpret_cast<uint32_t *>(raw_right[stage][r]) + word);
        }
        for (int32_t index = id; index < kSharedRows * kStageBlocks; index += kThreads) {
          const int32_t r = index / kStageBlocks;
          const int32_t part = index % kStageBlocks;
          const int32_t d = first + part * 32;
          StageWord<1>(reinterpret_cast<const uint8_t *>(
                           scales + static_cast<uint64_t>(row_start + r) * (inner / 32) + d / 32),
                       row_start + r < rows && d < stop_inner,
                       reinterpret_cast<uint32_t *>(&input_scales[stage][r][part]));
        }
      } else {
        for (int32_t index = id; index < kLoadRows * kStageBlocks * 8; index += kThreads) {
          const int32_t r = index / (kStageBlocks * 8);
          const int32_t local = index % (kStageBlocks * 8);
          const int32_t d = first + local * 4;
          if (r < kSharedRows) {
            left[stage][r][local] =
                row_start + r < rows && d < stop_inner
                    ? ActivationCodes<T, kIsPacked>(input,
                                                    static_cast<int64_t>(row_start + r) * inner + d)
                    : 0U;
          }
          if (r < kColumns) {
            right[r][local] = column_start + r < columns && d < stop_inner
                                  ? PackedWeightCodes<8>(
                                        weight, static_cast<uint64_t>(column_start + r) * inner + d)
                                  : 0U;
          }
        }
        for (int32_t index = id; index < kLoadRows * kStageBlocks; index += kThreads) {
          const int32_t r = index / kStageBlocks;
          const int32_t part = index % kStageBlocks;
          const int32_t d = first + part * 32;
          if (r < kSharedRows) {
            input_scales[stage][r][part] =
                row_start + r < rows && d < stop_inner
                    ? scales[static_cast<uint64_t>(row_start + r) * (inner / 32) + d / 32]
                    : 0.0F;
          }
          if (r < kColumns) {
            weight_scales[r][part] =
                column_start + r < columns && d < stop_inner
                    ? WeightScale<8>(weight, static_cast<uint64_t>(column_start + r) * inner + d)
                    : 0.0F;
          }
        }
      }
    };
    if constexpr (kIsAsync != 0) {
      stage_input(first_inner, 0);
      asm volatile("cp.async.commit_group;");
    }
    int32_t stage = 0;
    for (int32_t first = first_inner; first < stop_inner; first += kStageBlocks * 32) {
      if constexpr (kIsAsync != 0) {
        if (first + kStageBlocks * 32 < stop_inner) {
          stage_input(first + kStageBlocks * 32, stage ^ 1);
          asm volatile("cp.async.commit_group;");
          asm volatile("cp.async.wait_group 1;");
        } else {
          asm volatile("cp.async.wait_group 0;");
        }
      } else {
        stage_input(first, 0);
      }
      __syncthreads();
      for (int32_t part = 0; part < kStageBlocks && first + part * 32 < stop_inner; ++part) {
        uint32_t a[4];
        LoadLeft(reinterpret_cast<const uint16_t *>(&left[stage][0][0]), kStride * 2, warp_row / 16,
                 lane, part * 16, a);
#pragma unroll
        for (int32_t tile = 0; tile < kTiles; ++tile) {
          uint32_t b[2];
          if constexpr (kIsAsync != 0) {
            const uint8_t *const source = raw_right[stage][warp_column + tile * 8 + lane / 4];
            b[0] = PackedWeightCodes<8>(source, part * 32 + lane % 4 * 4);
            b[1] = PackedWeightCodes<8>(source, part * 32 + lane % 4 * 4 + 16);
          } else {
            LoadRight(reinterpret_cast<const uint16_t *>(&right[0][0]), kStride * 2,
                      tile + warp_column / 8, lane, part * 16, b);
          }
          int32_t partial[4];
          asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                       "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {0,0,0,0};"
                       : "=r"(partial[0]), "=r"(partial[1]), "=r"(partial[2]), "=r"(partial[3])
                       : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#pragma unroll
          for (int32_t item = 0; item < 4; ++item) {
            const int32_t row = warp_row + lane / 4 + item / 2 * 8;
            const int32_t column = warp_column + tile * 8 + lane % 4 * 2 + item % 2;
            const float weight_scale = kIsAsync != 0
                                           ? WeightScale<8>(raw_right[stage][column], part * 32)
                                           : weight_scales[column][part];
            const float weighted = static_cast<float>(partial[item]) * weight_scale;
            accumulators[tile][item] =
                fmaf(weighted, input_scales[stage][row][part], accumulators[tile][item]);
          }
        }
      }
      __syncthreads();
      if constexpr (kIsAsync != 0) {
        stage ^= 1;
      }
    }
    if constexpr (kIsSplitK != 0) {
#pragma unroll
      for (int32_t tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int32_t item = 0; item < 4; ++item) {
          if (!has_last) {
            last[tile][item] = accumulators[tile][item];
          } else {
            previous[tile][item] += accumulators[tile][item];
          }
        }
      }
      has_last = true;
    }
  }
#pragma unroll
  for (int32_t tile = 0; tile < kTiles; ++tile) {
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t local_row = warp_row + lane / 4 + item / 2 * 8;
      const int32_t row = row_start + local_row;
      const int32_t column = column_start + warp_column + tile * 8 + lane % 4 * 2 + item % 2;
      if (local_row < kRows && row < rows && column < columns) {
        const float sum =
            kIsSplitK != 0 ? last[tile][item] + previous[tile][item] : accumulators[tile][item];
        output[static_cast<uint64_t>(row) * columns + column] = FromFloat<T>(sum * global_scale);
      }
    }
  }
}

template <typename T>
__global__ void EmbeddingImpl(const int32_t *const indices, const uint8_t *const weight,
                              const int32_t weight_type, const int32_t tokens, const int32_t hidden,
                              const int32_t vocabulary, T *const output) {
  const uint64_t count = static_cast<uint64_t>(tokens) * hidden;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    const int32_t token = indices[index / hidden];
    output[index] =
        FromFloat<T>(token >= 0 && token < vocabulary
                         ? ReadWeight(weight, weight_type,
                                      static_cast<uint64_t>(token) * hidden + index % hidden)
                         : nanf(""));
  }
}

template <typename T>
__global__ void GemvImpl(const T *const input, const uint8_t *const weight,
                         const int32_t weight_type, const float weight_scale, const int32_t inner,
                         const int32_t columns, T *const output) {
  const int32_t lane = threadIdx.x % 32;
  const int32_t column = blockIdx.x * 8 + threadIdx.x / 32;
  if (column >= columns) {
    return;
  }
  float sum = 0.0F;
  for (int32_t d = lane; d < inner; d += 32) {
    sum = fmaf(ToFloat(input[d]),
               ReadWeight(weight, weight_type, static_cast<uint64_t>(column) * inner + d), sum);
  }
  sum = WarpReduceSum(sum);
  if (lane == 0) {
    output[column] = FromFloat<T>(sum * weight_scale);
  }
}

template <typename T, typename M>
__device__ void Matmul(const T *const input, const uint8_t *const weight, const int32_t weight_type,
                       const float weight_scale, const int32_t rows, const int32_t inner,
                       const int32_t columns, const float *const row_scales,
                       const int32_t scale_stride, T *const output) {
  __shared__ __align__(16) M left[64][40];
  __shared__ __align__(16) M right[64][40];
  const int32_t id = threadIdx.x;
  const int32_t lane = id % 32;
  const int32_t warp = id / 32;
  const int32_t row_start = blockIdx.y * 64;
  const int32_t column_start = blockIdx.x * 64;
  float accumulators[8][4] = {};
  for (int32_t first = 0; first < inner; first += 32) {
    for (int32_t index = id; index < 64 * 32; index += 128) {
      const int32_t r = index / 32;
      const int32_t d = first + index % 32;
      left[r][index % 32] =
          FromFloat<M>(row_start + r < rows && d < inner
                           ? ToFloat(input[static_cast<uint64_t>(row_start + r) * inner + d])
                           : 0.0F);
      right[r][index % 32] = FromFloat<M>(
          column_start + r < columns && d < inner
              ? ReadWeight(weight, weight_type, static_cast<uint64_t>(column_start + r) * inner + d)
              : 0.0F);
    }
    __syncthreads();
#pragma unroll
    for (int32_t d = 0; d < 32; d += 16) {
      uint32_t a[4];
      LoadLeft(&left[0][0], 40, warp, lane, d, a);
#pragma unroll
      for (int32_t tile = 0; tile < 8; ++tile) {
        uint32_t b[2];
        LoadRight(&right[0][0], 40, tile, lane, d, b);
        TensorMma<M>(a, b, accumulators[tile]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int32_t tile = 0; tile < 8; ++tile) {
#pragma unroll
    for (int32_t item = 0; item < 4; ++item) {
      const int32_t row = row_start + warp * 16 + lane / 4 + item / 2 * 8;
      const int32_t column = column_start + tile * 8 + (lane % 4) * 2 + item % 2;
      if (row < rows && column < columns) {
        output[static_cast<uint64_t>(row) * columns + column] = FromFloat<T>(
            (row_scales == nullptr ? accumulators[tile][item]
                                   : accumulators[tile][item] *
                                         row_scales[static_cast<uint64_t>(row) * scale_stride]) *
            weight_scale);
      }
    }
  }
}

template <typename T>
__global__ void MatmulImpl(const T *const input, const uint8_t *const weight,
                           const int32_t weight_type, const float weight_scale, const int32_t rows,
                           const int32_t inner, const int32_t columns, T *const output) {
  if (weight_type == 30 || weight_type == 40) {
    Matmul<T, __nv_bfloat16>(input, weight, weight_type, weight_scale, rows, inner, columns,
                             nullptr, 0, output);
  } else {
    Matmul<T, __half>(input, weight, weight_type, weight_scale, rows, inner, columns, nullptr, 0,
                      output);
  }
}

} // namespace omni_runtime::kernels::quantized_kernel
