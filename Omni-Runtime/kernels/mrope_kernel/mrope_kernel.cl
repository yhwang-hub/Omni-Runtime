inline int OmniPositionAxis(int pair, int temporal_section, int height_section, int width_section) {
  if (pair % 3 == 1 && pair < 3 * height_section) {
    return 1;
  }
  return pair % 3 == 2 && pair < 3 * width_section ? 2 : 0;
}

inline float OmniRotate(float first, float paired, int dimension, int head_dim, float position,
                        float frequency_base) {
  const int half_dim = head_dim / 2;
  const int pair = dimension % half_dim;
  const float exponent = -2.0f * (float)pair / (float)head_dim;
  const float angle = position * pow(frequency_base, exponent);
  const float cosine = RoundScalar(cos(angle));
  const float sine = RoundScalar(sin(angle));
  const float sign = dimension < half_dim ? -1.0f : 1.0f;
  return RoundScalar(RoundScalar(first * cosine) + RoundScalar(sign * paired * sine));
}

inline void OmniRotateGlobal(__global const scalar_t *input, __global scalar_t *output,
                             __global const int *positions, ulong index, int tokens, int heads,
                             int head_dim, int temporal_section, int height_section,
                             int width_section, float frequency_base) {
  if (index >= (ulong)tokens * heads * head_dim) {
    return;
  }
  const int dimension = index % head_dim;
  const int half_dim = head_dim / 2;
  const int paired = dimension < half_dim ? dimension + half_dim : dimension - half_dim;
  const int token = index / ((ulong)heads * head_dim);
  const int axis =
      OmniPositionAxis(dimension % half_dim, temporal_section, height_section, width_section);
  WriteScalar(output, index,
              OmniRotate(ReadScalar(input, index), ReadScalar(input, index - dimension + paired),
                         dimension, head_dim, (float)positions[(ulong)axis * tokens + token],
                         frequency_base));
}

// Flat max(query_elements, key_elements), rounded to a local size of 256.
__kernel void MultiAxisRope(__global uchar *query_buffer, ulong query_offset,
                            __global uchar *key_buffer, ulong key_offset,
                            __global uchar *position_ids_buffer, ulong position_ids_offset,
                            int tokens, int query_heads, int key_value_heads, int head_dim,
                            int temporal_section, int height_section, int width_section,
                            float frequency_base, __global uchar *query_output_buffer,
                            ulong query_output_offset, __global uchar *key_output_buffer,
                            ulong key_output_offset) {
  if (head_dim <= 0 || head_dim % 2 != 0 || temporal_section < 0 || height_section < 0 ||
      width_section < 0 || temporal_section + height_section + width_section != head_dim / 2 ||
      frequency_base <= 0.0f) {
    return;
  }
  const ulong index = get_global_id(0);
  OmniRotateGlobal(OMNI_SCALAR(query), OMNI_SCALAR(query_output), OMNI_INT(position_ids), index,
                   tokens, query_heads, head_dim, temporal_section, height_section, width_section,
                   frequency_base);
  OmniRotateGlobal(OMNI_SCALAR(key), OMNI_SCALAR(key_output), OMNI_INT(position_ids), index, tokens,
                   key_value_heads, head_dim, temporal_section, height_section, width_section,
                   frequency_base);
}

// One 256-work-item group per query/key head; even head_dim <= 256.
__kernel void NormalizeRopeCache(
    __global uchar *query_buffer, ulong query_offset, __global uchar *key_buffer, ulong key_offset,
    __global uchar *value_buffer, ulong value_offset, __global uchar *query_weight_buffer,
    ulong query_weight_offset, __global uchar *key_weight_buffer, ulong key_weight_offset,
    __global uchar *positions_buffer, ulong positions_offset, __global uchar *cache_position_buffer,
    ulong cache_position_offset, int query_heads, int key_value_heads, int head_dim,
    int cache_capacity, int temporal_section, int height_section, int width_section,
    float frequency_base, float epsilon, __global uchar *query_output_buffer,
    ulong query_output_offset, __global uchar *key_cache_buffer, ulong key_cache_offset,
    __global uchar *value_cache_buffer, ulong value_cache_offset) {
  const int head = get_group_id(0);
  if (head >= query_heads + key_value_heads || head_dim <= 0 || head_dim > 256 ||
      head_dim % 2 != 0) {
    return;
  }
  const int is_query = head < query_heads;
  const int local_head = is_query ? head : head - query_heads;
  const ulong first = (ulong)local_head * head_dim;
  __global const scalar_t *input = is_query ? OMNI_SCALAR(query) : OMNI_SCALAR(key);
  __global const scalar_t *weight = is_query ? OMNI_SCALAR(query_weight) : OMNI_SCALAR(key_weight);
  const int d = get_local_id(0);
  float square = 0.0f;
  if (d < head_dim) {
    const float value = ReadScalar(input, first + d);
    square = fma(value, value, square);
  }
  __local float scratch[256];
  __local float normalized[256];
  const float sum = OmniBlockSum(square, scratch);
  const float inverse = rsqrt(sum / (float)head_dim + epsilon);
  if (d < head_dim) {
    normalized[d] =
        RoundScalar(RoundScalar(ReadScalar(input, first + d) * inverse) * ReadScalar(weight, d));
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (d < head_dim) {
    const int half_dim = head_dim / 2;
    const int paired = d < half_dim ? d + half_dim : d - half_dim;
    const int axis =
        OmniPositionAxis(d % half_dim, temporal_section, height_section, width_section);
    const float rotated = OmniRotate(normalized[d], normalized[paired], d, head_dim,
                                     (float)OMNI_INT(positions)[axis], frequency_base);
    if (is_query) {
      WriteScalar(OMNI_SCALAR(query_output), first + d, rotated);
    } else {
      const int position = OMNI_INT(cache_position)[0];
      if (position >= 0 && position < cache_capacity) {
        const ulong destination = (ulong)position * key_value_heads * head_dim + first + d;
        WriteScalar(OMNI_SCALAR(key_cache), destination, rotated);
        OMNI_SCALAR(value_cache)[destination] = OMNI_SCALAR(value)[first + d];
      }
    }
  }
}
