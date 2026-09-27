#include <cstdint>
#include <mma.h>

#include <kernels/common/kernel_type_utils.h>

namespace omni_runtime {
namespace kernels {
namespace matmul_kernel {

constexpr int32_t kTile = 16;

__device__ void GemvHalfImpl(const __half *const __restrict__ input,
                             const __half *const __restrict__ weight, const int32_t inner,
                             const int32_t columns, __half *const __restrict__ output) {
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
  const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
  const int32_t warp_count = static_cast<int32_t>(blockDim.x) >> 5;
  const int32_t column = static_cast<int32_t>(blockIdx.x) * warp_count + warp;
  if (column >= columns || inner <= 0) {
    return;
  }
  const int64_t weight_offset = static_cast<int64_t>(column) * inner;
  float accumulator = 0.0f;
  if ((inner & 1) == 0) {
    constexpr int32_t kWarpHalfValuesPerIteration = 64;
    for (int32_t index = lane * 2; index < inner; index += kWarpHalfValuesPerIteration) {
      const float2 input_values = __half22float2(*reinterpret_cast<const __half2 *>(input + index));
      const float2 weight_values =
          __half22float2(*reinterpret_cast<const __half2 *>(weight + weight_offset + index));
      accumulator = fmaf(input_values.x, weight_values.x, accumulator);
      accumulator = fmaf(input_values.y, weight_values.y, accumulator);
    }
  } else {
    for (int32_t index = lane; index < inner; index += warpSize) {
      accumulator =
          fmaf(ToFloat(input[index]), ToFloat(weight[weight_offset + index]), accumulator);
    }
  }
  accumulator = WarpReduceSum(accumulator);
  if (lane == 0) {
    output[column] = FromFloat<__half>(accumulator);
  }
}

__device__ void MatmulFloatImpl(const float *const __restrict__ lhs,
                                const float *const __restrict__ rhs, const int32_t rows,
                                const int32_t inner, const int32_t columns,
                                float *const __restrict__ output) {
  __shared__ float lhs_tile[kTile][kTile + 1];
  __shared__ float rhs_tile[kTile][kTile + 1];
  const int32_t row = static_cast<int32_t>(blockIdx.y) * kTile + threadIdx.y;
  const int32_t column = static_cast<int32_t>(blockIdx.x) * kTile + threadIdx.x;
  float accumulator = 0.0f;
  for (int32_t tile_offset = 0; tile_offset < inner; tile_offset += kTile) {
    const int32_t lhs_column = tile_offset + threadIdx.x;
    const int32_t rhs_row = tile_offset + threadIdx.y;
    lhs_tile[threadIdx.y][threadIdx.x] = row < rows && lhs_column < inner
                                             ? lhs[static_cast<int64_t>(row) * inner + lhs_column]
                                             : 0.0f;
    rhs_tile[threadIdx.y][threadIdx.x] = rhs_row < inner && column < columns
                                             ? rhs[static_cast<int64_t>(rhs_row) * columns + column]
                                             : 0.0f;
    __syncthreads();
#pragma unroll
    for (int32_t index = 0; index < kTile; ++index) {
      accumulator = fmaf(lhs_tile[threadIdx.y][index], rhs_tile[index][threadIdx.x], accumulator);
    }
    __syncthreads();
  }
  if (row < rows && column < columns) {
    output[static_cast<int64_t>(row) * columns + column] = accumulator;
  }
}

template <typename T>
__device__ void MatmulTensorCoreImpl(const T *const __restrict__ lhs,
                                     const T *const __restrict__ rhs, const int32_t rows,
                                     const int32_t inner, const int32_t columns,
                                     T *const __restrict__ output) {
  using namespace nvcuda;
  __shared__ __align__(16) T lhs_tile[kTile * kTile];
  __shared__ __align__(16) T rhs_tile[kTile * kTile];
  __shared__ __align__(16) float output_tile[kTile * kTile];
  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  const int32_t row_start = static_cast<int32_t>(blockIdx.y) * kTile;
  const int32_t column_start = static_cast<int32_t>(blockIdx.x) * kTile;
  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, T, wmma::row_major> lhs_fragment;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, T, wmma::row_major> rhs_fragment;
  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (int32_t tile_offset = 0; tile_offset < inner; tile_offset += kTile) {
    for (int32_t index = lane; index < kTile * kTile; index += warpSize) {
      const int32_t tile_row = index / kTile;
      const int32_t tile_column = index - tile_row * kTile;
      const int32_t lhs_row = row_start + tile_row;
      const int32_t lhs_column = tile_offset + tile_column;
      const int32_t rhs_row = tile_offset + tile_row;
      const int32_t rhs_column = column_start + tile_column;
      lhs_tile[index] = lhs_row < rows && lhs_column < inner
                            ? lhs[static_cast<int64_t>(lhs_row) * inner + lhs_column]
                            : FromFloat<T>(0.0f);
      rhs_tile[index] = rhs_row < inner && rhs_column < columns
                            ? rhs[static_cast<int64_t>(rhs_row) * columns + rhs_column]
                            : FromFloat<T>(0.0f);
    }
    __syncwarp();
    wmma::load_matrix_sync(lhs_fragment, lhs_tile, kTile);
    wmma::load_matrix_sync(rhs_fragment, rhs_tile, kTile);
    wmma::mma_sync(accumulator, lhs_fragment, rhs_fragment, accumulator);
    __syncwarp();
  }
  wmma::store_matrix_sync(output_tile, accumulator, kTile, wmma::mem_row_major);
  __syncwarp();
  for (int32_t index = lane; index < kTile * kTile; index += warpSize) {
    const int32_t tile_row = index / kTile;
    const int32_t tile_column = index - tile_row * kTile;
    const int32_t row = row_start + tile_row;
    const int32_t column = column_start + tile_column;
    if (row < rows && column < columns) {
      output[static_cast<int64_t>(row) * columns + column] = FromFloat<T>(output_tile[index]);
    }
  }
}

} // namespace matmul_kernel
} // namespace kernels
} // namespace omni_runtime

