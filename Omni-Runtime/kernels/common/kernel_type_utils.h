#pragma once

#include <cfloat>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace omni_runtime {
namespace kernels {

template <typename T> __device__ __host__ inline float ToFloat(const T value) {
  return static_cast<float>(value);
}

template <> __device__ __host__ inline float ToFloat<__half>(const __half value) {
  return __half2float(value);
}

template <> __device__ __host__ inline float ToFloat<__nv_bfloat16>(const __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T> __device__ __host__ inline T FromFloat(const float value) {
  return static_cast<T>(value);
}

template <> __device__ __host__ inline __half FromFloat<__half>(const float value) {
  return __float2half_rn(value);
}

template <> __device__ __host__ inline __nv_bfloat16 FromFloat<__nv_bfloat16>(const float value) {
  return __float2bfloat16_rn(value);
}

#if defined(__CUDACC__)
__device__ inline float WarpReduceSum(float value) {
  for (int32_t offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__device__ inline float WarpReduceMax(float value) {
  for (int32_t offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ inline float BlockReduceSum(float value, float *const warp_values) {
  value = WarpReduceSum(value);
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
  const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
  if (lane == 0) {
    warp_values[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_values[lane] : 0.0f;
  if (warp == 0) {
    value = WarpReduceSum(value);
  }
  if (threadIdx.x == 0) {
    warp_values[0] = value;
  }
  __syncthreads();
  return warp_values[0];
}

__device__ inline float BlockReduceMax(float value, float *const warp_values) {
  value = WarpReduceMax(value);
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
  const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
  if (lane == 0) {
    warp_values[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_values[lane] : -FLT_MAX;
  if (warp == 0) {
    value = WarpReduceMax(value);
  }
  if (threadIdx.x == 0) {
    warp_values[0] = value;
  }
  __syncthreads();
  return warp_values[0];
}
#endif

} // namespace kernels
} // namespace omni_runtime
