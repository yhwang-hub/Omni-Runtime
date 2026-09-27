#define OMNI_DECODE_ARGUMENTS                                                                      \
  __global uchar *query_buffer, ulong query_offset, __global uchar *key_cache_buffer,              \
      ulong key_cache_offset, __global uchar *value_cache_buffer, ulong value_cache_offset,        \
      int query_token_count, int query_head_count, int key_value_head_count,                       \
      int cached_token_count, int start_position, int head_dim, int is_causal,                     \
      __global uchar *output_buffer, ulong output_offset, __global uchar *decode_partials_buffer,  \
      ulong decode_partials_offset, __global uchar *cache_position_buffer,                         \
      ulong cache_position_offset, int has_decode_partials, int has_cache_position

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
