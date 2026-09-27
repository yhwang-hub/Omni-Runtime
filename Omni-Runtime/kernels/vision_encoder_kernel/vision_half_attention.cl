#if defined(OMNI_NVIDIA_MMA)
#define VISION_HALF_QUERIES 64
#else
#define VISION_HALF_QUERIES 32
#endif

inline uint VisionHalfPair(float x, float y) {
  return (uint)OmniFloatToHalf(x) | ((uint)OmniFloatToHalf(y) << 16);
}

inline float VisionHalfExp(float value) {
#ifdef OMNI_NVIDIA_WARP
  const float exponent = value * 1.4426950408889634f;
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(exponent));
  return result;
#else
  return native_exp(value);
#endif
}

inline float2 VisionHalfFloats(uint value) {
  return (float2)(OmniHalfToFloat((ushort)value), OmniHalfToFloat((ushort)(value >> 16)));
}

inline uint VisionHalfMultiply(uint a, uint b) {
#ifdef OMNI_NVIDIA_WARP
  uint result;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
  return result;
#else
  const float2 result = VisionHalfFloats(a) * VisionHalfFloats(b);
  return VisionHalfPair(result.x, result.y);
#endif
}

inline uint VisionHalfAccumulate(uint value, uint probability, uint accumulator) {
#ifdef OMNI_NVIDIA_WARP
  uint result;
  asm("fma.rn.f16x2 %0, %1, %2, %3;"
      : "=r"(result)
      : "r"(value), "r"(probability), "r"(accumulator));
  return result;
#else
  const float2 result =
      fma(VisionHalfFloats(value), VisionHalfFloats(probability), VisionHalfFloats(accumulator));
  return VisionHalfPair(result.x, result.y);
#endif
}

inline float VisionWarpReduce(float value, int is_maximum, __local float *scratch) {
  const int id = get_local_id(0);
  for (int delta = 16; delta > 0; delta /= 2) {
    float other;
#ifdef OMNI_NVIDIA_WARP
    asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=f"(other) : "f"(value), "r"(delta));
#else
    scratch[id] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
    other = scratch[id ^ delta];
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    value = is_maximum ? fmax(value, other) : value + other;
  }
  return value;
}

inline void LoadVisionHalfTile(__global uchar *source, ulong source_offset, int tokens, int heads,
                               int head, int first, __local ushort *tile) {
  for (int index = get_local_id(0); index < 64 * 9; index += get_local_size(0)) {
    const int row = index / 9;
    const ulong offset = ((ulong)(first + row) * heads + head) * 72 + index % 9 * 8;
    if (first + row >= tokens) {
      vstore8((ushort8)0, index, tile);
    } else {
#ifdef OMNI_FP16_KEY_VALUE
      __global const uchar *values = source + source_offset;
      if (((ulong)values & 15U) == 0U) {
        *(__local uint4 *)(tile + index * 8) = *(__global const uint4 *)(values + offset * 2);
      } else {
        vstore8(vload8(offset / 8, (__global const ushort *)values), index, tile);
      }
#else
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        tile[index * 8 + item] = OmniFloatToHalf(
            ReadScalar((__global const scalar_t *)(source + source_offset), offset + item));
      }
#endif
    }
  }
}

__attribute__((reqd_work_group_size(VISION_HALF_QUERIES * 4, 1, 1))) __kernel void
VisionHalfAttention(__global uchar *query_buffer, ulong query_offset, __global uchar *key_buffer,
                    ulong key_offset, __global uchar *value_buffer, ulong value_offset, int tokens,
                    int heads, __global uchar *output_buffer, ulong output_offset) {
  const int id = get_local_id(0);
  const int lane = id % 32;
  const int warp = id / 32;
  const int tiles = (tokens + VISION_HALF_QUERIES - 1) / VISION_HALF_QUERIES;
  const int first_query = get_group_id(0) % tiles * VISION_HALF_QUERIES;
  const int head = get_group_id(0) / tiles;
  __local ushort queries[VISION_HALF_QUERIES][72] __attribute__((aligned(16)));
  __local ushort keys[64][72] __attribute__((aligned(16)));
  __local ushort probabilities[VISION_HALF_QUERIES / 8][64][8] __attribute__((aligned(16)));
  __local float scratch[VISION_HALF_QUERIES * 4];
  uint accumulators[2][8][2] = {{{0}}};
  float maxima[2][8];
  float denominators[2][8] = {{0}};
#pragma unroll
  for (int part = 0; part < 2; ++part) {
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      maxima[part][q] = -INFINITY;
    }
  }
  const float scale = 1.0f / sqrt(72.0f);
  for (int index = id; index < VISION_HALF_QUERIES * 72; index += VISION_HALF_QUERIES * 4) {
    const int row = index / 72;
    const int d = index % 72;
    const ulong offset = ((ulong)(first_query + row) * heads + head) * 72 + d;
    const ushort x = OmniFloatToHalf(
        first_query + row < tokens ? ReadScalar(OMNI_SCALAR(query), offset) * scale : 0.0f);
    queries[row][d] = OmniFloatToHalf(OmniHalfToFloat(x) * 0.25f);
  }
  for (int first = 0; first < tokens; first += 128) {
#pragma unroll
    for (int part = 0; part < 2; ++part) {
      const int key_first = first + part * 64;
      if (key_first < tokens) {
        LoadVisionHalfTile(key_buffer, key_offset, tokens, heads, head, key_first, &keys[0][0]);
        barrier(CLK_LOCAL_MEM_FENCE);
        float scores[8][2] = {{0}};
        for (int d = 0; d < 72; d += 8) {
          const uint4 packed_keys0 = as_uint4(vload8(0, keys[lane] + d));
          const uint4 packed_keys1 = as_uint4(vload8(0, keys[lane + 32] + d));
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            const int row = warp * 8 + q;
            const uint4 packed_query = as_uint4(vload8(0, queries[row] + d));
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
              const float2 first_product =
                  VisionHalfFloats(VisionHalfMultiply(packed_query[pair], packed_keys0[pair]));
              const float2 second_product =
                  VisionHalfFloats(VisionHalfMultiply(packed_query[pair], packed_keys1[pair]));
              scores[q][0] += first_product.x + first_product.y;
              scores[q][1] += second_product.x + second_product.y;
            }
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        LoadVisionHalfTile(value_buffer, value_offset, tokens, heads, head, key_first, &keys[0][0]);
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          float maximum = maxima[part][q];
          for (int item = 0; item < 2; ++item) {
            scores[q][item] =
                key_first + lane + item * 32 < tokens ? scores[q][item] * 4.0f : -INFINITY;
            maximum = fmax(maximum, scores[q][item] + 3.0f * 0.6931f);
          }
          maximum = VisionWarpReduce(maximum, 1, scratch);
          const float rescale = isfinite(maximum) ? VisionHalfExp(maxima[part][q] - maximum) : 0.0f;
          const uint rounded_rescale = VisionHalfPair(rescale, rescale);
          float probability_sum = 0.0f;
          for (int item = 0; item < 2; ++item) {
            const float probability =
                isfinite(maximum) ? VisionHalfExp(scores[q][item] - maximum) : 0.0f;
            probability_sum += probability;
            probabilities[warp][lane + item * 32][q] = OmniFloatToHalf(probability);
            accumulators[part][q][item] =
                VisionHalfMultiply(accumulators[part][q][item], rounded_rescale);
          }
          denominators[part][q] = fma(denominators[part][q], rescale, probability_sum);
          maxima[part][q] = maximum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int source = 0; source < 64; ++source) {
          const int d = lane * 4;
          if (d < 72) {
            const uint2 packed_value = as_uint2(vload4(0, keys[source] + d));
            const uint4 packed_probabilities = as_uint4(vload8(0, probabilities[warp][source]));
#pragma unroll
            for (int q = 0; q < 8; ++q) {
              const uint probability = (packed_probabilities[q / 2] >> ((q % 2) * 16)) & 0xFFFFU;
              const uint p = probability | (probability << 16);
#pragma unroll
              for (int item = 0; item < 2; ++item) {
                accumulators[part][q][item] =
                    VisionHalfAccumulate(packed_value[item], p, accumulators[part][q][item]);
              }
            }
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
  }
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int row = first_query + warp * 8 + q;
    const float maximum = fmax(maxima[0][q], maxima[1][q]);
    const float first_scale = VisionHalfExp(maxima[0][q] - maximum);
    const float second_scale = VisionHalfExp(maxima[1][q] - maximum);
    const float first_denominator = VisionWarpReduce(denominators[0][q], 0, scratch);
    const float second_denominator = VisionWarpReduce(denominators[1][q], 0, scratch);
    const float denominator =
        fma(second_scale, second_denominator, first_scale * first_denominator);
    for (int item = 0; item < 2; ++item) {
      const int d = lane * 4 + item * 2;
      if (row < tokens && d < 72) {
        const float2 a = VisionHalfFloats(accumulators[0][q][item]);
        const float2 b = VisionHalfFloats(accumulators[1][q][item]);
        const float2 result =
            native_divide(fma((float2)(second_scale), b, a * first_scale), (float2)(denominator));
        const ulong offset = ((ulong)row * heads + head) * 72 + d;
        WriteScalar(OMNI_SCALAR(output), offset, result.x);
        WriteScalar(OMNI_SCALAR(output), offset + 1, result.y);
      }
    }
  }
}
