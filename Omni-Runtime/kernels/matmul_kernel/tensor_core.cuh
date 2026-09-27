#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace omni_runtime::kernels {

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

template <typename T> __device__ inline uint32_t PackPair(const T first, const T second) {
  if constexpr (std::is_same_v<T, __half>) {
    return static_cast<uint32_t>(__half_as_ushort(first)) |
           (static_cast<uint32_t>(__half_as_ushort(second)) << 16U);
  } else {
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    return static_cast<uint32_t>(__bfloat16_as_ushort(first)) |
           (static_cast<uint32_t>(__bfloat16_as_ushort(second)) << 16U);
  }
}

// PTX m16n8k16: each four-lane group owns two rows and two columns of C.
template <typename T>
__device__ inline void TensorMma(const uint32_t (&a)[4], const uint32_t (&b)[2], float *const c) {
  if constexpr (std::is_same_v<T, __half>) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else {
    static_assert(std::is_same_v<T, __nv_bfloat16>);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
}

} // namespace omni_runtime::kernels
