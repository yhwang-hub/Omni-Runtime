#define OMNI_DECODE_ARGUMENTS                                                                      \
  __global uchar *query_buffer, ulong query_offset, __global uchar *key_cache_buffer,              \
      ulong key_cache_offset, __global uchar *value_cache_buffer, ulong value_cache_offset,        \
      int query_token_count, int query_head_count, int key_value_head_count,                       \
      int cached_token_count, int start_position, int head_dim, int is_causal,                     \
      __global uchar *output_buffer, ulong output_offset, __global uchar *decode_partials_buffer,  \
      ulong decode_partials_offset, __global uchar *cache_position_buffer,                         \
      ulong cache_position_offset, int has_decode_partials, int has_cache_position

#if defined(OMNI_GGML_SOFTMAX) && defined(OMNI_NVIDIA_MMA)
inline void LoadGgmlDecodeCache(__global uchar *source, ulong source_offset, int first, int count,
                                 int kv_heads, int head, __local ushort *cache) {
  for (int index = get_local_id(0); index < 64 * 64; index += 128) {
    const int token = first + index / 64;
    const int d = index % 64 * 4;
    const ulong offset = ((ulong)token * kv_heads + head) * 256 + d;
#ifdef OMNI_FP16_KEY_VALUE
    ushort4 packed = (ushort4)(0);
    if (token < count) {
      __global const ushort *input = (__global const ushort *)(source + source_offset);
      packed = vload4(0, input + offset);
    }
#else
    float4 values = (float4)(0.0f);
    if (token < count) {
      __global const float *input = (__global const float *)(source + source_offset);
      values = ((ulong)input & 15UL) == 0UL ? *(__global const float4 *)(input + offset)
                                            : vload4(0, input + offset);
    }
    const ushort4 packed = (ushort4)(OmniFloatToHalf(values.s0), OmniFloatToHalf(values.s1),
                                     OmniFloatToHalf(values.s2), OmniFloatToHalf(values.s3));
#endif
    *(__local uint2 *)(cache + index / 64 * 264 + d) = as_uint2(packed);
  }
}

__kernel void AttentionGgmlDecode(OMNI_DECODE_ARGUMENTS) {
  __global float *partials = (__global float *)(decode_partials_buffer + decode_partials_offset);
  const int id = get_local_id(0);
  const int lane = id % 32;
  const int warp = id / 32;
  const int row = lane / 4;
  const int column = (lane % 4) * 2;
  const int kv_head = get_group_id(0);
  const int part = get_group_id(1);
  const int ratio = query_head_count / key_value_head_count;
  const int position = has_cache_position ? OMNI_INT(cache_position)[0] : start_position;
  const int count = has_cache_position ? position + 1 : cached_token_count;
  const int limit = min(count, position + 1);
  const int iterations = ((count + 255) / 256) * 4;
  const int partitions = min(OMNI_GGML_PARTITION_CAP, iterations);
  if (part >= partitions) {
    return;
  }
  const int begin = part * iterations / partitions * 64;
  const int end = min((part + 1) * iterations / partitions * 64, limit);
  __local ushort cache[64 * 264] __attribute__((aligned(16)));
  __local ushort probabilities[64 * 8] __attribute__((aligned(16)));
  __local float metadata[4][8][2];
  __local float combined[8][2];
  uint2 queries[16];
  uint2 accumulators[16];
  float maxima[2] = {-FLT_MAX / 2.0f, -FLT_MAX / 2.0f};
  float denominators[2] = {0.0f, 0.0f};
#pragma unroll
  for (int d = 0; d < 256; d += 16) {
    accumulators[d / 16] = (uint2)(0U);
    const int head = kv_head * ratio + row;
    const ulong offset = (ulong)head * 256 + d + column;
#pragma unroll
    for (int item = 0; item < 2; ++item) {
      queries[d / 16][item] =
          row < ratio
              ? OmniPack(OmniFloatToHalf(OmniHalfToFloat(OmniFloatToHalf(
                             ReadScalar(OMNI_SCALAR(query), offset + item * 8))) * 0.0625f),
                         OmniFloatToHalf(OmniHalfToFloat(OmniFloatToHalf(
                             ReadScalar(OMNI_SCALAR(query), offset + item * 8 + 1))) * 0.0625f))
              : 0U;
    }
  }
  for (int first = begin; first < end; first += 64) {
    LoadGgmlDecodeCache(key_cache_buffer, key_cache_offset, first, count, key_value_head_count,
                        kv_head, cache);
    barrier(CLK_LOCAL_MEM_FENCE);
    float scores[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int d = 0; d < 256; d += 16) {
      const uint address =
          (uint)(ulong)(cache + (warp * 16 + lane % 16) * 264 + d + lane / 16 * 8);
      uint a[4];
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                   : "r"(address));
      const uint b[2] = {queries[d / 16].s0, queries[d / 16].s1};
      OmniMma(a, b, scores);
    }
    float rescales[2];
#pragma unroll
    for (int item = 0; item < 2; ++item) {
      float maximum = maxima[item];
#pragma unroll
      for (int source = 0; source < 2; ++source) {
        const int token = first + warp * 16 + row + source * 8;
        scores[source * 2 + item] =
            token < limit && column + item < ratio ? scores[source * 2 + item] : -INFINITY;
        maximum = fmax(maximum, scores[source * 2 + item] + 3.0f * 0.6931f);
      }
#pragma unroll
      for (int distance = 16; distance >= 4; distance >>= 1) {
        maximum = fmax(maximum, OmniShuffleXor(maximum, distance));
      }
      const float difference = maxima[item] - maximum;
      rescales[item] = difference >= -20.0f ? native_exp(difference) : 0.0f;
      float sum = 0.0f;
#pragma unroll
      for (int source = 0; source < 2; ++source) {
        const float probability = native_exp(scores[source * 2 + item] - maximum);
        const int token = warp * 16 + row + source * 8;
        probabilities[token * 8 + column + item] = OmniFloatToHalf(probability);
        sum += probability;
      }
      denominators[item] = fma(denominators[item], rescales[item], sum);
      maxima[item] = maximum;
    }
#pragma unroll
    for (int d = 0; d < 16; ++d) {
#pragma unroll
      for (int item = 0; item < 2; ++item) {
        const ushort2 previous = as_ushort2(accumulators[d][item]);
        accumulators[d][item] =
            OmniPack(OmniFloatToHalf(OmniHalfToFloat(previous.s0) *
                                      OmniHalfToFloat(OmniFloatToHalf(rescales[0]))),
                     OmniFloatToHalf(OmniHalfToFloat(previous.s1) *
                                      OmniHalfToFloat(OmniFloatToHalf(rescales[1]))));
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    LoadGgmlDecodeCache(value_cache_buffer, value_cache_offset, first, count, key_value_head_count,
                        kv_head, cache);
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint probability_address = (uint)(ulong)(probabilities + (warp * 16 + lane % 16) * 8);
    uint2 b;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b.s0), "=r"(b.s1)
                 : "r"(probability_address));
#pragma unroll
    for (int d = 0; d < 256; d += 16) {
      const uint address = (uint)(ulong)(cache + (warp * 16 + lane % 8 + lane / 16 * 8) * 264 +
                                         d + (lane / 8 % 2) * 8);
      uint4 a;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a.s0), "=r"(a.s1), "=r"(a.s2), "=r"(a.s3)
                   : "r"(address));
      accumulators[d / 16] = OmniTensorMmaHalf(a, b, accumulators[d / 16]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#pragma unroll
  for (int item = 0; item < 2; ++item) {
#pragma unroll
    for (int distance = 16; distance >= 4; distance >>= 1) {
      denominators[item] += OmniShuffleXor(denominators[item], distance);
    }
    if (row == 0) {
      metadata[warp][column + item][0] = maxima[item];
      metadata[warp][column + item][1] = denominators[item];
    }
  }
#pragma unroll
  for (int d = 0; d < 256; d += 16) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int head = column + item % 2;
      const int dimension = d + row + item / 2 * 8;
      const uint bits = accumulators[d / 16][item / 2] >> (item % 2 * 16);
      cache[(warp * 8 + head) * 256 + dimension] = (ushort)bits;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (id < 8) {
    float maximum = metadata[0][id][0];
#pragma unroll
    for (int stream = 1; stream < 4; ++stream) {
      maximum = fmax(maximum, metadata[stream][id][0]);
    }
    float sums[4];
#pragma unroll
    for (int stream = 0; stream < 4; ++stream) {
      const float scale = native_exp(metadata[stream][id][0] - maximum);
      asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(sums[stream])
                   : "f"(scale), "f"(metadata[stream][id][1]));
      metadata[stream][id][0] = scale;
    }
    combined[id][0] = maximum;
    combined[id][1] = (sums[0] + sums[2]) + (sums[1] + sums[3]);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int index = id; index < ratio * 256; index += 128) {
    const int head = index / 256;
    const int d = index % 256;
    float sum = 0.0f;
#pragma unroll
    for (int stream = 0; stream < 4; ++stream) {
      sum = fma(OmniHalfToFloat(cache[(stream * 8 + head) * 256 + d]),
                 metadata[stream][head][0], sum);
    }
    const ulong offset = ((ulong)part * query_head_count + kv_head * ratio + head) * 258;
    partials[offset + d] = sum;
    if (d == 0) {
      partials[offset + 256] = combined[head][0];
      partials[offset + 257] = combined[head][1];
    }
  }
}

__kernel void AttentionGgmlDecodeMerge(OMNI_DECODE_ARGUMENTS) {
  __global const float *partials =
      (__global const float *)(decode_partials_buffer + decode_partials_offset);
  const int count = has_cache_position ? OMNI_INT(cache_position)[0] + 1 : cached_token_count;
  const int partitions = min(OMNI_GGML_PARTITION_CAP, ((count + 255) / 256) * 4);
  const int head = get_group_id(0);
  const int d = get_local_id(0);
  ulong offset = ((ulong)(partitions - 1) * query_head_count + head) * 258;
  float numerator = partials[offset + d];
  float maximum = partials[offset + 256];
  float denominator = partials[offset + 257];
  for (int part = partitions - 2; part >= 0; --part) {
    offset = ((ulong)part * query_head_count + head) * 258;
    const float next = fmax(maximum, partials[offset + 256]);
    const float difference = maximum - next;
    const float addition_difference = partials[offset + 256] - next;
    const float scale = difference >= -20.0f ? native_exp(difference) : 0.0f;
    const float addition_scale = addition_difference >= -20.0f ? native_exp(addition_difference) : 0.0f;
    numerator = fma(scale, numerator, addition_scale * partials[offset + d]);
    denominator = fma(scale, denominator, addition_scale * partials[offset + 257]);
    maximum = next;
  }
  WriteScalar(OMNI_SCALAR(output), (ulong)head * 256 + d, native_divide(numerator, denominator));
}
#endif

__kernel void AttentionDecode(OMNI_DECODE_ARGUMENTS) {
  const int head = get_group_id(0);
  const int partition = get_group_id(1);
  const int first = partition * 128;
  const int count = has_cache_position ? OMNI_INT(cache_position)[0] + 1 : cached_token_count;
  if (first >= count) {
    return;
  }
  const int limit = min(count, first + 128);
  const int id = get_local_id(0);
  const int lane = id % 8;
  const int kv_head = head / (query_head_count / key_value_head_count);
  __local float dots[128][8];
  __local float scores[128];
  __local float scratch[128];
  for (int index = id / 8; index < 128; index += 16) {
    const int token = first + index;
    const ulong offset = ((ulong)token * key_value_head_count + kv_head) * head_dim;
    float score = 0.0f;
    for (int d = lane; d < head_dim; d += 8) {
      if (token < limit) {
        score = fma(ReadScalar(OMNI_SCALAR(query), (ulong)head * head_dim + d),
                    ReadScalar(OMNI_SCALAR(key_cache), offset + d), score);
      }
    }
#ifdef OMNI_NVIDIA_WARP
    for (int delta = 4; delta > 0; delta >>= 1) {
      float other;
      asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=f"(other) : "f"(score), "r"(delta));
      score += other;
    }
    if (lane == 0) {
      scores[index] = token < limit ? score * rsqrt((float)head_dim) : -INFINITY;
    }
#else
    dots[index][lane] = score;
#endif
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#ifndef OMNI_NVIDIA_WARP
  const float even = (dots[id][0] + dots[id][4]) + (dots[id][2] + dots[id][6]);
  const float odd = (dots[id][1] + dots[id][5]) + (dots[id][3] + dots[id][7]);
  scores[id] = first + id < limit ? (even + odd) * rsqrt((float)head_dim) : -INFINITY;
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
  scratch[id] = scores[id];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int delta = 64; delta > 0; delta >>= 1) {
    if (id < delta) {
      scratch[id] = fmax(scratch[id], scratch[id + delta]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float maximum = scratch[0];
  const float probability = exp(scores[id] - maximum);
  barrier(CLK_LOCAL_MEM_FENCE);
  const float denominator = OmniBlockSum(probability, scratch);
  scores[id] = probability;
  barrier(CLK_LOCAL_MEM_FENCE);
  __global float *partials = (__global float *)(decode_partials_buffer + decode_partials_offset);
  const ulong output_first = ((ulong)partition * query_head_count + head) * (head_dim + 2);
  for (int d = id; d < head_dim; d += 128) {
    float accumulator = 0.0f;
    for (int token = first; token < limit; ++token) {
      const ulong offset = ((ulong)token * key_value_head_count + kv_head) * head_dim + d;
      accumulator =
          fma(scores[token - first], ReadScalar(OMNI_SCALAR(value_cache), offset), accumulator);
    }
    partials[output_first + d] = accumulator;
  }
  if (id == 0) {
    partials[output_first + head_dim] = maximum;
    partials[output_first + head_dim + 1] = denominator;
  }
}

__kernel void AttentionDecodeMerge(OMNI_DECODE_ARGUMENTS) {
  const int head = get_group_id(0);
  const int d = get_local_id(0);
  if (d >= head_dim) {
    return;
  }
  const int count = has_cache_position ? OMNI_INT(cache_position)[0] + 1 : cached_token_count;
  const int partitions = (count + 127) / 128;
  __global const float *partials =
      (__global const float *)(decode_partials_buffer + decode_partials_offset);
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float accumulator = 0.0f;
  for (int partition = 0; partition < partitions; ++partition) {
    const ulong offset = ((ulong)partition * query_head_count + head) * (head_dim + 2);
    const float next = fmax(maximum, partials[offset + head_dim]);
    const float rescale = exp(maximum - next);
    const float weight = exp(partials[offset + head_dim] - next);
    denominator = denominator * rescale + partials[offset + head_dim + 1] * weight;
    accumulator = accumulator * rescale + partials[offset + d] * weight;
    maximum = next;
  }
  WriteScalar(OMNI_SCALAR(output), (ulong)head * head_dim + d, accumulator / denominator);
}

#if defined(OMNI_NVIDIA_MMA) && OMNI_DTYPE != 0
#include "gqa_attention_kernel/decode_grouped.cl"
#endif
