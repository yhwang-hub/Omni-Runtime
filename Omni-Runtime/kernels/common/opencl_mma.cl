#ifndef OMNI_OPENCL_MMA
#define OMNI_OPENCL_MMA

#if defined(OMNI_NVIDIA_MMA) && (OMNI_DTYPE == 1 || OMNI_DTYPE == 2 || defined(OMNI_HALF_ATTENTION))
inline uint OmniPack(ushort first, ushort second) {
  return (uint)first | ((uint)second << 16);
}

inline ushort OmniScalarBits(float value) {
#if OMNI_DTYPE != 2
  return OmniFloatToHalf(value);
#else
  return OmniFloatToBfloat(value);
#endif
}

inline float4 OmniTensorMma(uint4 a, uint2 b, float4 accumulator) {
  float c0 = accumulator.s0;
  float c1 = accumulator.s1;
  float c2 = accumulator.s2;
  float c3 = accumulator.s3;
#if OMNI_DTYPE != 2
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a.s0), "r"(a.s1), "r"(a.s2), "r"(a.s3), "r"(b.s0), "r"(b.s1));
#else
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a.s0), "r"(a.s1), "r"(a.s2), "r"(a.s3), "r"(b.s0), "r"(b.s1));
#endif
  return (float4)(c0, c1, c2, c3);
}

inline void OmniMma(__private const uint *a, __private const uint *b, __private float *c) {
  const float4 value =
      OmniTensorMma((uint4)(a[0], a[1], a[2], a[3]), (uint2)(b[0], b[1]), vload4(0, c));
  vstore4(value, 0, c);
}

#ifdef OMNI_GGML_SOFTMAX
inline uint2 OmniTensorMmaHalf(uint4 a, uint2 b, uint2 accumulator) {
  uint c0 = accumulator.s0;
  uint c1 = accumulator.s1;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
               : "+r"(c0), "+r"(c1)
               : "r"(a.s0), "r"(a.s1), "r"(a.s2), "r"(a.s3), "r"(b.s0), "r"(b.s1));
  return (uint2)(c0, c1);
}

inline void OmniMmaHalf(__private const uint *a, __private const uint *b, __private float *c) {
  const uint2 result = OmniTensorMmaHalf(
      (uint4)(a[0], a[1], a[2], a[3]), (uint2)(b[0], b[1]),
      (uint2)(OmniPack(OmniFloatToHalf(c[0]), OmniFloatToHalf(c[1])), 0U));
  const ushort2 first = as_ushort2(result.s0);
  c[0] = OmniHalfToFloat(first.s0);
  c[1] = OmniHalfToFloat(first.s1);
}
#endif

#if OMNI_DTYPE != 0
inline ushort2 OmniLoadScalarPair(__global const scalar_t *buffer, ulong index, int remaining) {
  if (remaining >= 2) {
    // vload permits scalar-aligned BufferViews and odd matrix row strides.
    return vload2(0, buffer + index);
  }
  return (ushort2)(remaining > 0 ? buffer[index] : (ushort)0, (ushort)0);
}
#endif

inline uint OmniLocalPair(__local const ushort *buffer, int index) {
  return as_uint(vload2(0, buffer + index));
}

inline uint4 OmniTensorLeft(__local const ushort *tile, int warp, int lane, int d) {
  const int row = warp * 16 + lane / 4;
  const int column = d + (lane % 4) * 2;
  return (uint4)(OmniLocalPair(tile, row * 40 + column),
                 OmniLocalPair(tile, (row + 8) * 40 + column),
                 OmniLocalPair(tile, row * 40 + column + 8),
                 OmniLocalPair(tile, (row + 8) * 40 + column + 8));
}

inline uint2 OmniTensorRight(__local const ushort *tile, int output_tile, int lane, int d) {
  const int index = (output_tile * 8 + lane / 4) * 40 + d + (lane % 4) * 2;
  return (uint2)(OmniLocalPair(tile, index), OmniLocalPair(tile, index + 8));
}

inline float OmniPackedScalar(uint bits) {
#if OMNI_DTYPE != 2
  return OmniHalfToFloat((ushort)bits);
#else
  return as_float(bits << 16);
#endif
}
#endif

#if defined(OMNI_NVIDIA_WARP)
inline float OmniShuffleXor(float value, int delta) {
  float other;
  asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=f"(other) : "f"(value), "r"(delta));
  return other;
}

inline float OmniWarpSum(float value) {
  for (uint delta = 16; delta != 0; delta >>= 1) {
    float other;
    asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=f"(other) : "f"(value), "r"(delta));
    value += other;
  }
  return value;
}
#endif

#endif
