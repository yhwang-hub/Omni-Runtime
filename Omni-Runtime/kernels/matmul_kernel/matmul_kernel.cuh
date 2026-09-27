#pragma once
#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace matmul_kernel {

constexpr int32_t kTile = 16;

template <bool kStreaming>
__device__ __forceinline__ uint4 LoadGemvWeight(const void *const address) {
#if __CUDA_ARCH__ >= 1200
  if constexpr (kStreaming) {
    uint4 value;
    asm volatile("ld.global.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(address));
    return value;
  }
#endif
  return *static_cast<const uint4 *>(address);
}

template <typename T, bool kVectorized, int32_t kInner, bool kStreaming>
__device__ __forceinline__ float GemvDot(const T *const input, const T *const weight,
                                         const int32_t inner, const int32_t lane) {
  constexpr int32_t kValues = 16 / sizeof(T);
  const int32_t count = kInner == 0 ? inner : kInner;
  float accumulators[kValues] = {};
  if constexpr (kVectorized) {
#pragma unroll 4
    for (int32_t column = lane * kValues; column < count; column += 32 * kValues) {
      const uint4 packed_input = *reinterpret_cast<const uint4 *>(input + column);
      const uint4 packed_weight = LoadGemvWeight<kStreaming>(weight + column);
      const T *const inputs = reinterpret_cast<const T *>(&packed_input);
      const T *const weights = reinterpret_cast<const T *>(&packed_weight);
#pragma unroll
      for (int32_t item = 0; item < kValues; ++item) {
        accumulators[item] =
            fmaf(ToFloat(inputs[item]), ToFloat(weights[item]), accumulators[item]);
      }
    }
  } else {
    for (int32_t column = lane; column < count; column += 32) {
      accumulators[0] = fmaf(ToFloat(input[column]), ToFloat(weight[column]), accumulators[0]);
    }
  }
  float sum = 0.0F;
#pragma unroll
  for (int32_t item = 0; item < kValues; ++item) {
    sum += accumulators[item];
  }
  return WarpReduceSum(sum);
}

template <typename T, bool kVectorized>
__global__ void LinearGemvImpl(const T *const input, const T *const weight, const int32_t inner,
                               const int32_t columns, T *const output) {
  const int32_t lane = threadIdx.x % 32;
  const int32_t row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= columns) {
    return;
  }
  const float sum = GemvDot<T, kVectorized, 0, false>(
      input, weight + static_cast<int64_t>(row) * inner, inner, lane);
  if (lane == 0) {
    output[row] = FromFloat<T>(sum);
  }
}

template <typename T, int32_t kInner>
__global__ __launch_bounds__(256) void BlackwellGemvImpl(const T *const input,
                                                         const T *const weight,
                                                         const int32_t columns, T *const output) {
  const int32_t lane = threadIdx.x % 32;
  const int32_t row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= columns) {
    return;
  }
  const float sum = GemvDot<T, true, kInner, true>(
      input, weight + static_cast<int64_t>(row) * kInner, kInner, lane);
  if (lane == 0) {
    output[row] = FromFloat<T>(sum);
  }
}

template <typename T, bool kVectorized, int32_t kInner>
__global__ __launch_bounds__(256) void ParallelGemvImpl(
    const T *const input, const T *const first_weight, const int32_t first_columns,
    T *const first_output, const T *const second_weight, const int32_t second_columns,
    T *const second_output, const T *const third_weight, const int32_t third_columns,
    T *const third_output, const int32_t inner) {
  const int32_t first_blocks = first_columns / 8 + (first_columns % 8 != 0);
  const int32_t second_blocks = second_columns / 8 + (second_columns % 8 != 0);
  const bool is_first = blockIdx.x < first_blocks;
  const bool is_second = !is_first && blockIdx.x < first_blocks + second_blocks;
  const int32_t block = is_first    ? blockIdx.x
                        : is_second ? blockIdx.x - first_blocks
                                    : blockIdx.x - first_blocks - second_blocks;
  const T *const weight = is_first ? first_weight : is_second ? second_weight : third_weight;
  T *const output = is_first ? first_output : is_second ? second_output : third_output;
  const int32_t columns = is_first ? first_columns : is_second ? second_columns : third_columns;
  const int32_t row = block * 8 + threadIdx.x / 32;
  if (row >= columns) {
    return;
  }
  const int32_t lane = threadIdx.x % 32;
  const T *const neuron = weight + static_cast<int64_t>(row) * inner;
  float sum = 0.0F;
  if constexpr (kInner != 0) {
    sum = columns <= 12288 ? GemvDot<T, true, kInner, true>(input, neuron, inner, lane)
                           : GemvDot<T, true, 0, false>(input, neuron, inner, lane);
  } else {
    sum = GemvDot<T, kVectorized, 0, false>(input, neuron, inner, lane);
  }
  if (lane == 0) {
    output[row] = FromFloat<T>(sum);
  }
}

template <typename T, bool kTransposed, int32_t kColumns>
__device__ void RunMatmul(const T *const lhs, const T *const rhs, const int32_t rows,
                          const int32_t inner, const int32_t columns, T *const output) {
  constexpr int32_t kOutputs = kColumns / kTile;
  __shared__ float lhs_tile[kTile][kTile + 1];
  __shared__ float rhs_tile[kTile][kColumns + 1];
  const int32_t row = static_cast<int32_t>(blockIdx.y) * kTile + threadIdx.y;
  const int32_t first_column = static_cast<int32_t>(blockIdx.x) * kColumns + threadIdx.x;
  float accumulators[kOutputs] = {};
  for (int32_t tile_offset = 0; tile_offset < inner; tile_offset += kTile) {
    const int32_t lhs_column = tile_offset + threadIdx.x;
    const int32_t rhs_row = tile_offset + threadIdx.y;
    lhs_tile[threadIdx.y][threadIdx.x] =
        row < rows && lhs_column < inner
            ? ToFloat(lhs[static_cast<int64_t>(row) * inner + lhs_column])
            : 0.0f;
#pragma unroll
    for (int32_t item = 0; item < kOutputs; ++item) {
      const int32_t column = first_column + item * kTile;
      const int64_t offset = kTransposed ? static_cast<int64_t>(column) * inner + rhs_row
                                         : static_cast<int64_t>(rhs_row) * columns + column;
      rhs_tile[threadIdx.y][threadIdx.x + item * kTile] =
          rhs_row < inner && column < columns ? ToFloat(rhs[offset]) : 0.0F;
    }
    __syncthreads();
#pragma unroll
    for (int32_t index = 0; index < kTile; ++index) {
#pragma unroll
      for (int32_t item = 0; item < kOutputs; ++item) {
        accumulators[item] = fmaf(lhs_tile[threadIdx.y][index],
                                  rhs_tile[index][threadIdx.x + item * kTile], accumulators[item]);
      }
    }
    __syncthreads();
  }
  if (row < rows) {
#pragma unroll
    for (int32_t item = 0; item < kOutputs; ++item) {
      const int32_t column = first_column + item * kTile;
      if (column < columns) {
        output[static_cast<int64_t>(row) * columns + column] = FromFloat<T>(accumulators[item]);
      }
    }
  }
}

template <typename T, int32_t kColumns>
__global__ void MatmulImpl(const T *const __restrict__ lhs, const T *const __restrict__ rhs,
                           const int32_t rows, const int32_t inner, const int32_t columns,
                           T *const __restrict__ output) {
  RunMatmul<T, false, kColumns>(lhs, rhs, rows, inner, columns, output);
}

template <typename T, int32_t kColumns>
__global__ void LinearMatmulImpl(const T *const __restrict__ lhs, const T *const __restrict__ rhs,
                                 const int32_t rows, const int32_t inner, const int32_t columns,
                                 T *const __restrict__ output) {
  RunMatmul<T, true, kColumns>(lhs, rhs, rows, inner, columns, output);
}

} // namespace matmul_kernel
} // namespace kernels
} // namespace omni_runtime
