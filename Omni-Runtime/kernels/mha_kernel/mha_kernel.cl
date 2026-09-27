// Head-major [head, token, dimension], with the CUDA 16-key online-softmax
// recurrence. Scores and probabilities remain FP32 for all storage types.
__kernel void MultiHeadAttention(__global uchar *query_buffer, ulong query_offset,
                                 __global uchar *key_buffer, ulong key_offset,
                                 __global uchar *value_buffer, ulong value_offset, int heads,
                                 int tokens, int head_dim, int is_causal,
                                 __global uchar *output_buffer, ulong output_offset) {
  const ulong row = get_group_id(0);
  if (row >= (ulong)heads * tokens) {
    return;
  }
  const int token = row % tokens;
  const ulong head = row / tokens;
  const uint id = get_local_id(0);
  const ulong query_first = row * head_dim;
  const ulong head_first = head * tokens * head_dim;
  __local float scores[16];
  __local float probabilities[16];
  __local float scratch[16];
  float accumulators[8] = {0.0f};
  float maximum = -INFINITY;
  float denominator = 0.0f;
  const float scale = rsqrt((float)head_dim);
  const long limit = is_causal ? (long)token + 1 : tokens;
  for (long first = 0; first < limit; first += 16) {
    const long source = first + id;
    if (id < 16) {
      float even = 0.0f;
      float odd = 0.0f;
      if (source < limit) {
        const ulong key_first = head_first + source * head_dim;
        for (int d = 0; d < head_dim; d += 2) {
          even = fma(ReadScalar(OMNI_SCALAR(query), query_first + d),
                     ReadScalar(OMNI_SCALAR(key), key_first + d), even);
        }
        for (int d = 1; d < head_dim; d += 2) {
          odd = fma(ReadScalar(OMNI_SCALAR(query), query_first + d),
                    ReadScalar(OMNI_SCALAR(key), key_first + d), odd);
        }
      }
      scores[id] = source < limit ? (even + odd) * scale : -INFINITY;
      scratch[id] = fmax(maximum, scores[id]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint delta = 8; delta != 0; delta >>= 1) {
      if (id < delta) {
        scratch[id] = fmax(scratch[id], scratch[id + delta]);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float next_maximum = scratch[0];
    const float rescale = exp(maximum - next_maximum);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (id < 16) {
      probabilities[id] = exp(scores[id] - next_maximum);
      scratch[id] = probabilities[id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint delta = 8; delta != 0; delta >>= 1) {
      if (id < delta) {
        scratch[id] += scratch[id + delta];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    denominator = fma(denominator, rescale, scratch[0]);
    maximum = next_maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int item = 0; item < 8; ++item) {
      const int d = id + item * 32;
      if (d < head_dim) {
        float weighted = 0.0f;
        for (int k = 0; k < 16; ++k) {
          const long key_token = first + k;
          if (key_token < tokens) {
            weighted = fma(probabilities[k],
                           ReadScalar(OMNI_SCALAR(value), head_first + key_token * head_dim + d),
                           weighted);
          }
        }
        accumulators[item] = fma(accumulators[item], rescale, weighted);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int item = 0; item < 8; ++item) {
    const int d = id + item * 32;
    if (d < head_dim) {
      WriteScalar(OMNI_SCALAR(output), query_first + d, accumulators[item] / denominator);
    }
  }
}
