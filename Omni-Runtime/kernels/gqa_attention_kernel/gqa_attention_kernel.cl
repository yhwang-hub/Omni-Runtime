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

#if defined(OMNI_NVIDIA_MMA) && OMNI_DTYPE != 0
#ifndef OMNI_ATTENTION_DIM
#define OMNI_ATTENTION_DIM 128
#endif
#define OMNI_ATTENTION_KEYS 64
#if OMNI_ATTENTION_DIM <= 80
#define OMNI_ATTENTION_QUERIES 128
#else
#define OMNI_ATTENTION_QUERIES 64
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
  float accumulators[OMNI_ATTENTION_DIM / 8][4] = {{0.0f}};
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
          token < query_token_count && dimension < head_dim
              ? OmniPack(OMNI_SCALAR(query)[offset],
                         dimension + 1 < head_dim ? OMNI_SCALAR(query)[offset + 1] : (ushort)0)
              : 0U;
    }
  }
  const int limit =
      is_causal ? min((long)cached_token_count,
                      (long)start_position + ((long)get_group_id(0) + 1) * OMNI_ATTENTION_QUERIES)
                : cached_token_count;
  const float scale = rsqrt((float)head_dim);
  for (int first_key = 0; first_key < limit; first_key += OMNI_ATTENTION_KEYS) {
    for (int index = id * 8; index < OMNI_ATTENTION_KEYS * OMNI_ATTENTION_DIM;
         index += get_local_size(0) * 8) {
      const int token = first_key + index / OMNI_ATTENTION_DIM;
      const int dimension = index % OMNI_ATTENTION_DIM;
      const ulong offset = ((ulong)token * key_value_head_count + kv_head) * head_dim + dimension;
      const int target = index / OMNI_ATTENTION_DIM * OMNI_ATTENTION_STRIDE + dimension;
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const int is_valid = token < cached_token_count && dimension + item < head_dim;
        keys[target + item] = is_valid ? OMNI_SCALAR(key_cache)[offset + item] : (ushort)0;
        values[target + item] = is_valid ? OMNI_SCALAR(value_cache)[offset + item] : (ushort)0;
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
    for (int item_row = 0; item_row < 2; ++item_row) {
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
                                      ? scores[tile][element] * scale
                                      : -INFINITY;
          maximum = fmax(maximum, scores[tile][element]);
        }
      }
      maximum = fmax(maximum, OmniShuffleXor(maximum, 1));
      maximum = fmax(maximum, OmniShuffleXor(maximum, 2));
      const float rescale = isfinite(maximum) ? exp(maxima[item_row] - maximum) : 0.0f;
      float sum = 0.0f;
#pragma unroll
      for (int tile = 0; tile < OMNI_ATTENTION_KEYS / 8; ++tile) {
#pragma unroll
        for (int item = 0; item < 2; ++item) {
          const int element = item_row * 2 + item;
          const float probability = isfinite(maximum) ? exp(scores[tile][element] - maximum) : 0.0f;
          scores[tile][element] = probability;
          sum += probability;
        }
      }
      sum += OmniShuffleXor(sum, 1);
      sum += OmniShuffleXor(sum, 2);
      denominators[item_row] = denominators[item_row] * rescale + sum;
      maxima[item_row] = maximum;
#pragma unroll
      for (int tile = 0; tile < OMNI_ATTENTION_DIM / 8; ++tile) {
        accumulators[tile][item_row * 2] *= rescale;
        accumulators[tile][item_row * 2 + 1] *= rescale;
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
        OmniMma(a, b, accumulators[tile]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#pragma unroll
  for (int tile = 0; tile < OMNI_ATTENTION_DIM / 8; ++tile) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int token = first_query + row + (item / 2) * 8;
      const int d = tile * 8 + column + item % 2;
      if (token < query_token_count && d < head_dim) {
        WriteScalar(OMNI_SCALAR(output), ((ulong)token * query_head_count + head) * head_dim + d,
                    accumulators[tile][item] / denominators[item / 2]);
      }
    }
  }
}
#endif
