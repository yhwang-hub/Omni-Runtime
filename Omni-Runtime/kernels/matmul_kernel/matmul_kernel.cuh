#pragma once
#include <cstdint>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace matmul_kernel {

constexpr int32_t kTile = 16;

template <typename T, bool kVectorized>
__global__ void LinearGemvImpl(const T *const input, const T *const weight, const int32_t inner,
                               const int32_t columns, T *const output) {
  constexpr int32_t kValues = 16 / sizeof(T);
  const int32_t lane = threadIdx.x % 32;
  const int32_t row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= columns) {
    return;
  }
  float accumulators[kValues] = {};
  if constexpr (kVectorized) {
    for (int32_t column = lane * kValues; column < inner; column += 32 * kValues) {
      const uint4 packed_input = *reinterpret_cast<const uint4 *>(input + column);
      const uint4 packed_weight =
          *reinterpret_cast<const uint4 *>(weight + static_cast<int64_t>(row) * inner + column);
      const T *const inputs = reinterpret_cast<const T *>(&packed_input);
      const T *const weights = reinterpret_cast<const T *>(&packed_weight);
#pragma unroll
      for (int32_t item = 0; item < kValues; ++item) {
        accumulators[item] =
            fmaf(ToFloat(inputs[item]), ToFloat(weights[item]), accumulators[item]);
      }
    }
  } else {
    for (int32_t column = lane; column < inner; column += 32) {
      accumulators[0] =
          fmaf(ToFloat(input[column]), ToFloat(weight[static_cast<int64_t>(row) * inner + column]),
               accumulators[0]);
    }
  }
  float sum = 0.0F;
#pragma unroll
  for (int32_t item = 0; item < kValues; ++item) {
    sum += accumulators[item];
  }
  sum = WarpReduceSum(sum);
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
