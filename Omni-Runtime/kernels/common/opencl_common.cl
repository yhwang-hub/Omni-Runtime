#ifndef OMNI_OPENCL_COMMON
#define OMNI_OPENCL_COMMON

// Concatenate before each operator source; build as strict OpenCL C 1.2.
// Operand offsets are bytes; ReadScalar/WriteScalar indices are scalar elements.
#pragma OPENCL FP_CONTRACT OFF

#ifndef OMNI_DTYPE
#define OMNI_DTYPE 0
#endif

#if OMNI_DTYPE == 0
typedef float scalar_t;
#elif OMNI_DTYPE == 1 || OMNI_DTYPE == 2
typedef ushort scalar_t;
#else
#error Unsupported OMNI_DTYPE
#endif

// Integer conversions do not require cl_khr_fp16, including for subnormals.
inline ushort OmniFloatToHalf(float value) {
#ifdef OMNI_NVIDIA_WARP
  ushort result;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
  return result;
#else
  const uint bits = as_uint(value);
  const uint sign = (bits >> 16) & 0x8000U;
  const uint exponent_bits = (bits >> 23) & 255U;
  const uint fraction = bits & 0x7fffffU;
  if (exponent_bits == 255U) {
    return (ushort)(fraction ? 0x7fffU : sign | 0x7c00U);
  }
  const int exponent = (int)exponent_bits - 127;
  if (exponent > 15) {
    return (ushort)(sign | 0x7c00U);
  }
  if (exponent < -25) {
    return (ushort)sign;
  }
  const uint mantissa = fraction | 0x800000U;
  const uint shift = exponent < -14 ? (uint)(-exponent - 1) : 13U;
  uint rounded = mantissa >> shift;
  const uint remainder = mantissa & ((1U << shift) - 1U);
  const uint midpoint = 1U << (shift - 1U);
  rounded += remainder > midpoint || (remainder == midpoint && (rounded & 1U));
  return (ushort)(sign | (exponent < -14 ? rounded : (uint)((exponent + 14) << 10) + rounded));
#endif
}

inline float OmniHalfToFloat(ushort value) {
#ifdef OMNI_NVIDIA_WARP
  float result;
  asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
  return result;
#else
  const uint sign = ((uint)value & 0x8000U) << 16;
  const uint exponent = ((uint)value >> 10) & 31U;
  const uint fraction = (uint)value & 1023U;
  if (exponent == 0U) {
    const float magnitude = (float)fraction * 0x1.0p-24f;
    return as_float(as_uint(magnitude) | sign);
  }
  return as_float(sign | (exponent == 31U ? 0x7f800000U : (exponent + 112U) << 23) |
                  (fraction << 13));
#endif
}

inline ushort OmniFloatToBfloat(float value) {
#ifdef OMNI_NVIDIA_MMA
  ushort result;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(value));
  return result;
#else
  const uint bits = as_uint(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return (ushort)0x7fffU;
  }
  return (ushort)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
#endif
}

inline float RoundScalar(float value) {
#if OMNI_DTYPE == 1
  return OmniHalfToFloat(OmniFloatToHalf(value));
#elif OMNI_DTYPE == 2
  return as_float((uint)OmniFloatToBfloat(value) << 16);
#else
  return value;
#endif
}

inline float ReadScalar(__global const scalar_t *buffer, ulong index) {
#if OMNI_DTYPE == 1
  return OmniHalfToFloat(buffer[index]);
#elif OMNI_DTYPE == 2
  return as_float((uint)buffer[index] << 16);
#else
  return buffer[index];
#endif
}

inline void WriteScalar(__global scalar_t *buffer, ulong index, float value) {
#if OMNI_DTYPE == 1
  buffer[index] = OmniFloatToHalf(value);
#elif OMNI_DTYPE == 2
  buffer[index] = OmniFloatToBfloat(value);
#else
  buffer[index] = value;
#endif
}

#define OMNI_SCALAR(NAME) ((__global scalar_t *)(NAME##_buffer + NAME##_offset))
#define OMNI_INT(NAME) ((__global int *)(NAME##_buffer + NAME##_offset))

// Match CUDA's 32-lane shuffle-down tree, then its warp-leader tree.
// All work-items must call this with a 128- or 256-work-item group.
inline float OmniBlockSum(float value, __local float *scratch) {
  const uint id = get_local_id(0);
  const uint lane = id & 31U;
#ifdef OMNI_NVIDIA_WARP
  for (uint delta = 16; delta != 0; delta >>= 1) {
    float other;
    asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=f"(other) : "f"(value), "r"(delta));
    value += other;
  }
  if (lane == 0U) {
    scratch[id / 32U] = value;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  value = id < get_local_size(0) / 32U ? scratch[id] : 0.0f;
  for (uint delta = 16; delta != 0; delta >>= 1) {
    float other;
    asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=f"(other) : "f"(value), "r"(delta));
    value += other;
  }
  if (id == 0U) {
    scratch[0] = value;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  value = scratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return value;
#else
  scratch[id] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint delta = 16; delta != 0; delta >>= 1) {
    if (lane < delta) {
      scratch[id] += scratch[id + delta];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  value = id < get_local_size(0) / 32 ? scratch[id * 32] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  scratch[id] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint delta = 16; delta != 0; delta >>= 1) {
    if (id < delta) {
      scratch[id] += scratch[id + delta];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  value = scratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return value;
#endif
}

#endif
