// One 128-work-item group per flattened token * query_head_count + head.
// head_dim <= 256; 6156 bytes fixed local storage, independent of cache length.
// decode_partials is accepted but not touched: this entry computes direct attention.
__kernel void Attention(__global uchar *query_buffer, ulong query_offset,
                        __global uchar *key_cache_buffer, ulong key_cache_offset,
                        __global uchar *value_cache_buffer, ulong value_cache_offset,
                        int query_token_count, int query_head_count, int key_value_head_count,
                        int cached_token_count, int start_position, int head_dim, int is_causal,
                        __global uchar *output_buffer, ulong output_offset,
                        __global uchar *decode_partials_buffer, ulong decode_partials_offset,
                        __global uchar *cache_position_buffer, ulong cache_position_offset,
                        int has_decode_partials, int has_cache_position) {
  const ulong row = get_group_id(0);
  if (row >= (ulong)query_token_count * query_head_count || head_dim <= 0 || head_dim > 256 ||
      key_value_head_count <= 0 || query_head_count % key_value_head_count != 0) {
    return;
  }

  const int id = get_local_id(0);
  const int token = row / query_head_count;
  const int head = row % query_head_count;
  const int kv_head = head / (query_head_count / key_value_head_count);
  const int position = has_cache_position ? OMNI_INT(cache_position)[0] : start_position;
  const int count = has_cache_position ? position + 1 : cached_token_count;
  const long causal_limit = (long)position + token + 1;
  const int limit = is_causal ? (int)min((long)count, causal_limit) : count;
  const ulong query_first = row * head_dim;
  __local float queries[256];
  __local float dot_partials[128][8];
  __local float probabilities[128];
  __local float reduction[128];
  __local float maximum;
  __local float denominator;
  __local float rescale;
  for (int d = id; d < head_dim; d += 128) {
    queries[d] = ReadScalar(OMNI_SCALAR(query), query_first + d);
  }
  if (id == 0) {
    maximum = -INFINITY;
    denominator = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float accumulators[2] = {0.0f, 0.0f};
  const float scale = rsqrt((float)head_dim);
  // Fixed-size online softmax: no context-length or quadratic scratch.
  for (int first = 0; first < limit; first += 128) {
    const int dimension_lane = id % 8;
    const int first_key = id / 8;
    // Adjacent work-items read adjacent dimensions, including D=80 ViT heads.
    for (int item = 0; item < 8; ++item) {
      const int local_key = first_key + item * 16;
      const int source = first + local_key;
      float partial = 0.0f;
      if (source < limit) {
        const ulong key_first = ((ulong)source * key_value_head_count + kv_head) * head_dim;
        for (int d = dimension_lane; d < head_dim; d += 8) {
          partial = fma(queries[d], ReadScalar(OMNI_SCALAR(key_cache), key_first + d), partial);
        }
      }
      dot_partials[local_key][dimension_lane] = partial;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float even =
        (dot_partials[id][0] + dot_partials[id][4]) + (dot_partials[id][2] + dot_partials[id][6]);
    const float odd =
        (dot_partials[id][1] + dot_partials[id][5]) + (dot_partials[id][3] + dot_partials[id][7]);
    const float score = first + id < limit ? (even + odd) * scale : -INFINITY;
    probabilities[id] = score;
    reduction[id] = fmax(maximum, score);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int delta = 64; delta != 0; delta >>= 1) {
      if (id < delta) {
        reduction[id] = fmax(reduction[id], reduction[id + delta]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (id == 0) {
      const float next = reduction[0];
      rescale = isfinite(next) ? exp(maximum - next) : 0.0f;
      maximum = next;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float probability = isfinite(maximum) ? exp(probabilities[id] - maximum) : 0.0f;
    probabilities[id] = probability;
    reduction[id] = probability;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int delta = 64; delta != 0; delta >>= 1) {
      if (id < delta) {
        reduction[id] += reduction[id + delta];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (id == 0) {
      denominator = denominator * rescale + reduction[0];
    }
    for (int item = 0; item < 2; ++item) {
      const int d = id + item * 128;
      if (d < head_dim) {
        float weighted = 0.0f;
        for (int key = 0; key < 128 && first + key < limit; ++key) {
          const ulong value_first =
              ((ulong)(first + key) * key_value_head_count + kv_head) * head_dim;
          weighted += probabilities[key] * ReadScalar(OMNI_SCALAR(value_cache), value_first + d);
        }
        accumulators[item] = accumulators[item] * rescale + weighted;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int item = 0; item < 2; ++item) {
    const int d = id + item * 128;
    if (d < head_dim) {
      WriteScalar(OMNI_SCALAR(output), query_first + d, accumulators[item] / denominator);
    }
  }
}

#if defined(OMNI_NVIDIA_MMA) && (OMNI_DTYPE != 0 || defined(OMNI_HALF_ATTENTION))
inline ushort AttentionBits(scalar_t value) {
#if OMNI_DTYPE == 0
  return OmniFloatToHalf(value);
#else
  return value;
#endif
}
inline ushort AttentionCacheBits(__global uchar *buffer, ulong offset, ulong index) {
#ifdef OMNI_FP16_KEY_VALUE
  return ((__global ushort *)(buffer + offset))[index];
#else
  return AttentionBits(((__global scalar_t *)(buffer + offset))[index]);
#endif
}
#ifndef OMNI_ATTENTION_DIM
#define OMNI_ATTENTION_DIM 128
#endif
#if OMNI_ATTENTION_DIM > 128
#define OMNI_ATTENTION_KEYS 32
#else
#define OMNI_ATTENTION_KEYS 64
#endif
#ifdef OMNI_GGML_SOFTMAX
#define OMNI_ATTENTION_QUERIES 8
#define OMNI_ATTENTION_ITEMS 2
#define OMNI_ATTENTION_ROWS 1
#else
#define OMNI_ATTENTION_ITEMS 4
#define OMNI_ATTENTION_ROWS 2
#if OMNI_ATTENTION_DIM <= 80
#define OMNI_ATTENTION_QUERIES 128
#else
#define OMNI_ATTENTION_QUERIES 64
#endif
#endif
#define OMNI_ATTENTION_STRIDE (OMNI_ATTENTION_DIM + 8)

__kernel void AttentionTensor(__global uchar *query_buffer, ulong query_offset,
                              __global uchar *key_cache_buffer, ulong key_cache_offset,
                              __global uchar *value_cache_buffer, ulong value_cache_offset,
                              int query_token_count, int query_head_count, int key_value_head_count,
                              int cached_token_count, int start_position, int head_dim,
                              int is_causal, __global uchar *output_buffer, ulong output_offset,
                              __global uchar *decode_partials_buffer, ulong decode_partials_offset,
                              __global uchar *cache_position_buffer, ulong cache_position_offset,
                              int has_decode_partials, int has_cache_position) {
  if (has_cache_position) {
    start_position = OMNI_INT(cache_position)[0];
    cached_token_count = start_position + 1;
  }
  const int id = get_local_id(0);
  const int lane = id % 32;
  const int warp = id / 32;
  const int row = lane / 4;
  const int column = (lane % 4) * 2;
  const int head = get_group_id(1);
  const int kv_head = head / (query_head_count / key_value_head_count);
  const int first_query = get_group_id(0) * OMNI_ATTENTION_QUERIES + warp * 16;
  __local ushort keys[OMNI_ATTENTION_KEYS * OMNI_ATTENTION_STRIDE];
  __local ushort values[OMNI_ATTENTION_KEYS * OMNI_ATTENTION_STRIDE];
  uint queries[OMNI_ATTENTION_DIM / 16][4];
  float accumulators[OMNI_ATTENTION_DIM / 8][OMNI_ATTENTION_ITEMS] = {{0.0f}};
#ifdef OMNI_GGML_SOFTMAX
  float merged[OMNI_ATTENTION_DIM / 8][OMNI_ATTENTION_ITEMS] = {{0.0f}};
  float merged_maximum = -INFINITY;
  float merged_denominator = 0.0f;
#endif
  float maxima[2] = {-INFINITY, -INFINITY};
  float denominators[2] = {0.0f, 0.0f};
#pragma unroll
  for (int d = 0; d < OMNI_ATTENTION_DIM; d += 16) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int token = first_query + row + (item % 2) * 8;
      const int dimension = d + column + (item / 2) * 8;
      const ulong offset = ((ulong)token * query_head_count + head) * head_dim + dimension;
      queries[d / 16][item] =
          token < query_token_count && token < first_query + OMNI_ATTENTION_QUERIES &&
                  dimension < head_dim
              ? OmniPack(AttentionBits(OMNI_SCALAR(query)[offset]),
                         dimension + 1 < head_dim ? AttentionBits(OMNI_SCALAR(query)[offset + 1])
                                                  : (ushort)0)
              : 0U;
#ifdef OMNI_GGML_SOFTMAX
      const ushort2 pair = as_ushort2(queries[d / 16][item]);
      queries[d / 16][item] = OmniPack(OmniFloatToHalf(OmniHalfToFloat(pair.s0) * 0.0625f),
                                       OmniFloatToHalf(OmniHalfToFloat(pair.s1) * 0.0625f));
#endif
    }
  }
  const int limit =
      is_causal ? min((long)cached_token_count,
                      (long)start_position + ((long)get_group_id(0) + 1) * OMNI_ATTENTION_QUERIES)
                : cached_token_count;
  const float scale = rsqrt((float)head_dim);
#ifdef OMNI_GGML_SOFTMAX
#if OMNI_GGML_STREAM_GROUPS > 0
  const long token_tiles = ((long)query_token_count + 7) / 8;
  const long key_tiles = (((long)cached_token_count + 255) / 256) * 8;
  const long output_tile = kv_head * token_tiles + get_group_id(0);
  const long total_work = token_tiles * key_value_head_count * key_tiles;
  const int first_part = (int)(output_tile * key_tiles * OMNI_GGML_STREAM_GROUPS / total_work);
  const int last_part =
      (int)min((long)(OMNI_GGML_STREAM_GROUPS - 1),
               (output_tile + 1) * key_tiles * OMNI_GGML_STREAM_GROUPS / total_work);
#else
  const int first_part = 0;
  const int last_part = 0;
#endif
  for (int part = last_part; part >= first_part; --part) {
    int first_key_begin = 0;
    int first_key_stop = limit;
#if OMNI_GGML_STREAM_GROUPS > 0
    const long first = (long)part * total_work / OMNI_GGML_STREAM_GROUPS;
    const long stop = (long)(part + 1) * total_work / OMNI_GGML_STREAM_GROUPS;
    const long begin = max(first, output_tile * key_tiles);
    const long end = min(stop, (output_tile + 1) * key_tiles);
    if (begin >= end) {
      continue;
    }
    first_key_begin = (begin - output_tile * key_tiles) * OMNI_ATTENTION_KEYS;
    first_key_stop = min((end - output_tile * key_tiles) * OMNI_ATTENTION_KEYS, (long)limit);
#endif
    maxima[0] = -INFINITY;
    denominators[0] = 0.0f;
#pragma unroll
    for (int tile = 0; tile < OMNI_ATTENTION_DIM / 8; ++tile) {
#pragma unroll
      for (int item = 0; item < OMNI_ATTENTION_ITEMS; ++item) {
        accumulators[tile][item] = 0.0f;
      }
    }
#else
  const int first_key_begin = 0;
  const int first_key_stop = limit;
#endif
    for (int first_key = first_key_begin; first_key < first_key_stop;
         first_key += OMNI_ATTENTION_KEYS) {
      for (int index = id * 8; index < OMNI_ATTENTION_KEYS * OMNI_ATTENTION_DIM;
           index += get_local_size(0) * 8) {
        const int token = first_key + index / OMNI_ATTENTION_DIM;
        const int dimension = index % OMNI_ATTENTION_DIM;
        const ulong offset = ((ulong)token * key_value_head_count + kv_head) * head_dim + dimension;
        const int target = index / OMNI_ATTENTION_DIM * OMNI_ATTENTION_STRIDE + dimension;
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const int is_valid = token < cached_token_count && dimension + item < head_dim;
        keys[target + item] =
            is_valid ? AttentionCacheBits(key_cache_buffer, key_cache_offset, offset + item)
                     : (ushort)0;
        values[target + item] =
            is_valid ? AttentionCacheBits(value_cache_buffer, value_cache_offset, offset + item)
                     : (ushort)0;
      }
      }
    barrier(CLK_LOCAL_MEM_FENCE);
    float scores[OMNI_ATTENTION_KEYS / 8][4] = {{0.0f}};
#pragma unroll
    for (int d = 0; d < OMNI_ATTENTION_DIM; d += 16) {
#pragma unroll
      for (int tile = 0; tile < OMNI_ATTENTION_KEYS / 8; ++tile) {
        const int offset = (tile * 8 + row) * OMNI_ATTENTION_STRIDE + d + column;
        const uint b[2] = {OmniPack(keys[offset], keys[offset + 1]),
                           OmniPack(keys[offset + 8], keys[offset + 9])};
        OmniMma(queries[d / 16], b, scores[tile]);
      }
    }
#pragma unroll
    for (int item_row = 0; item_row < OMNI_ATTENTION_ROWS; ++item_row) {
      const int token = first_query + row + item_row * 8;
      float maximum = maxima[item_row];
#pragma unroll
      for (int tile = 0; tile < OMNI_ATTENTION_KEYS / 8; ++tile) {
#pragma unroll
        for (int item = 0; item < 2; ++item) {
          const int source = first_key + tile * 8 + column + item;
          const int element = item_row * 2 + item;
          scores[tile][element] = token < query_token_count && source < cached_token_count &&
                                          (!is_causal || source <= start_position + token)
                                      ? scores[tile][element]
#ifndef OMNI_GGML_SOFTMAX
                                            * scale
#endif
                                      : -INFINITY;
          maximum = fmax(maximum, scores[tile][element]
#ifdef OMNI_GGML_SOFTMAX
                                      + 3.0f * 0.6931f
#endif
          );
        }
      }
      maximum = fmax(maximum, OmniShuffleXor(maximum, 1));
      maximum = fmax(maximum, OmniShuffleXor(maximum, 2));
#ifdef OMNI_GGML_SOFTMAX
      const float rescale = isfinite(maximum) ? native_exp(maxima[item_row] - maximum) : 0.0f;
#else
      const float rescale = isfinite(maximum) ? exp(maxima[item_row] - maximum) : 0.0f;
#endif
      float sum = 0.0f;
#pragma unroll
      for (int tile = 0; tile < OMNI_ATTENTION_KEYS / 8; ++tile) {
#pragma unroll
        for (int item = 0; item < 2; ++item) {
          const int element = item_row * 2 + item;
          const float probability =
#ifdef OMNI_GGML_SOFTMAX
              isfinite(maximum) ? native_exp(scores[tile][element] - maximum) : 0.0f;
#else
              isfinite(maximum) ? exp(scores[tile][element] - maximum) : 0.0f;
#endif
          scores[tile][element] = probability;
          sum += probability;
        }
      }
#ifndef OMNI_GGML_SOFTMAX
      sum += OmniShuffleXor(sum, 1);
      sum += OmniShuffleXor(sum, 2);
#endif
#ifdef OMNI_GGML_SOFTMAX
      denominators[item_row] = fma(denominators[item_row], rescale, sum);
#else
      denominators[item_row] = denominators[item_row] * rescale + sum;
#endif
      maxima[item_row] = maximum;
#pragma unroll
      for (int tile = 0; tile < OMNI_ATTENTION_DIM / 8; ++tile) {
#ifdef OMNI_GGML_SOFTMAX
        const float rounded_rescale = OmniHalfToFloat(OmniFloatToHalf(rescale));
        accumulators[tile][item_row * 2] =
            OmniHalfToFloat(OmniFloatToHalf(accumulators[tile][item_row * 2] * rounded_rescale));
        accumulators[tile][item_row * 2 + 1] = OmniHalfToFloat(
            OmniFloatToHalf(accumulators[tile][item_row * 2 + 1] * rounded_rescale));
#else
        accumulators[tile][item_row * 2] *= rescale;
        accumulators[tile][item_row * 2 + 1] *= rescale;
#endif
      }
    }
#pragma unroll
    for (int source = 0; source < OMNI_ATTENTION_KEYS; source += 16) {
      uint a[4];
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int tile = source / 8 + item / 2;
        const int offset = (item % 2) * 2;
        a[item] = OmniPack(OmniScalarBits(scores[tile][offset]),
                           OmniScalarBits(scores[tile][offset + 1]));
      }
#pragma unroll
      for (int tile = 0; tile < OMNI_ATTENTION_DIM / 8; ++tile) {
        const int d = tile * 8 + row;
        const uint b[2] = {OmniPack(values[(source + column) * OMNI_ATTENTION_STRIDE + d],
                                    values[(source + column + 1) * OMNI_ATTENTION_STRIDE + d]),
                           OmniPack(values[(source + column + 8) * OMNI_ATTENTION_STRIDE + d],
                                    values[(source + column + 9) * OMNI_ATTENTION_STRIDE + d])};
#ifdef OMNI_GGML_SOFTMAX
        OmniMmaHalf(a, b, accumulators[tile]);
#else
        OmniMma(a, b, accumulators[tile]);
#endif
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    }
#ifdef OMNI_GGML_SOFTMAX
#pragma unroll
  for (int item = 0; item < OMNI_ATTENTION_ROWS; ++item) {
    denominators[item] += OmniShuffleXor(denominators[item], 2);
    denominators[item] += OmniShuffleXor(denominators[item], 1);
  }
  const float maximum = fmax(merged_maximum, maxima[0]);
  const float old_diff = merged_maximum - maximum;
  const float new_diff = maxima[0] - maximum;
  const float old_scale = old_diff >= -20.0f ? native_exp(old_diff) : 0.0f;
  const float new_scale = new_diff >= -20.0f ? native_exp(new_diff) : 0.0f;
  merged_denominator = fma(old_scale, merged_denominator, new_scale * denominators[0]);
  merged_maximum = maximum;
#pragma unroll
  for (int tile = 0; tile < OMNI_ATTENTION_DIM / 8; ++tile) {
#pragma unroll
    for (int item = 0; item < OMNI_ATTENTION_ITEMS; ++item) {
      merged[tile][item] = fma(old_scale, merged[tile][item], new_scale * accumulators[tile][item]);
    }
  }
  }
#endif
#pragma unroll
  for (int tile = 0; tile < OMNI_ATTENTION_DIM / 8; ++tile) {
#pragma unroll
    for (int item = 0; item < OMNI_ATTENTION_ITEMS; ++item) {
      const int token = first_query + row + (item / 2) * 8;
      const int d = tile * 8 + column + item % 2;
      if (token < query_token_count && d < head_dim) {
        WriteScalar(OMNI_SCALAR(output), ((ulong)token * query_head_count + head) * head_dim + d,
#ifdef OMNI_GGML_SOFTMAX
                    native_divide(merged[tile][item], merged_denominator));
#else
                    accumulators[tile][item] / denominators[item / 2]);
#endif
      }
    }
  }
}
#endif
