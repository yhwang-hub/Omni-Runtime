#pragma OPENCL FP_CONTRACT OFF

__kernel void CausalConv1d(__global uchar *input_buffer, ulong input_offset,
                           __global uchar *weight_buffer, ulong weight_offset,
                           __global uchar *state_buffer, ulong state_offset, int token_count,
                           int channels, int kernel_size, __global uchar *output_buffer,
                           ulong output_offset) {
  const int channel = get_global_id(0);
  if (channel >= channels) {
    return;
  }
  __global float *state = (__global float *)(state_buffer + state_offset);
  __global const float *weight = (__global const float *)(weight_buffer + weight_offset);
  float history[15];
  for (int j = 0; j < kernel_size - 1; ++j) {
    history[j] = state[(ulong)channel * (kernel_size - 1) + j];
  }
  for (int t = 0; t < token_count; ++t) {
    const ulong index = (ulong)t * channels + channel;
    const float current = ReadScalar(OMNI_SCALAR(input), index);
    float sum = 0.0f;
    for (int j = 0; j < kernel_size - 1; ++j) {
      sum = fma(history[j], weight[channel * kernel_size + j], sum);
    }
    sum = fma(current, weight[channel * kernel_size + kernel_size - 1], sum);
    WriteScalar(OMNI_SCALAR(output), index, native_divide(sum, 1.0f + native_exp(-sum)));
    for (int j = 0; j < kernel_size - 2; ++j) {
      history[j] = history[j + 1];
    }
    if (kernel_size > 1) {
      history[kernel_size - 2] = current;
    }
  }
  for (int j = 0; j < kernel_size - 1; ++j) {
    state[(ulong)channel * (kernel_size - 1) + j] = history[j];
  }
}

inline float DeltaWarpSum(float value, __local float *scratch) {
#ifdef OMNI_NVIDIA_WARP
  for (int distance = 16; distance > 0; distance >>= 1) {
    float other;
    asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=f"(other) : "f"(value), "r"(distance));
    value += other;
  }
  return value;
#else
  const uint id = get_local_id(0);
  scratch[id] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint distance = 16; distance > 0; distance >>= 1) {
    const float other = scratch[id ^ distance];
    barrier(CLK_LOCAL_MEM_FENCE);
    scratch[id] += other;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  value = scratch[id];
  barrier(CLK_LOCAL_MEM_FENCE);
  return value;
#endif
}

inline float DeltaNormSquareSum(const float *values, __local float *scratch) {
  float warps[4];
#pragma unroll
  for (int part = 0; part < 4; ++part) {
    warps[part] = DeltaWarpSum(values[part] * values[part], scratch);
  }
  return (warps[0] + warps[2]) + (warps[1] + warps[3]);
}

__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void
GatedDeltaNet(__global uchar *qkv_buffer, ulong qkv_offset, __global uchar *alpha_buffer,
              ulong alpha_offset, __global uchar *beta_buffer, ulong beta_offset,
              __global uchar *decay_weight_buffer, ulong decay_weight_offset,
              __global uchar *time_bias_buffer, ulong time_bias_offset,
              __global uchar *state_buffer, ulong state_offset, int token_count, int key_head_count,
              int value_head_count, int head_size, float epsilon, __global uchar *output_buffer,
              ulong output_offset) {
  const int head = get_group_id(0);
  const int column = get_group_id(1) * 4 + get_local_id(0) / 32;
  const int lane = get_local_id(0) % 32;
  const int is_valid = column < head_size;
  const int key_head = head % key_head_count;
  const int key_size = key_head_count * head_size;
  const int value_size = value_head_count * head_size;
  const int channels = 2 * key_size + value_size;
  __global float *state = (__global float *)(state_buffer + state_offset);
  __global const float *decay_weight =
      (__global const float *)(decay_weight_buffer + decay_weight_offset);
  __global const float *time_bias = (__global const float *)(time_bias_buffer + time_bias_offset);
  __local float scratch[128];
  float matrix[4];
  for (int j = 0; j < 4; ++j) {
    const int d = j * 32 + lane;
    matrix[j] = is_valid && d < head_size
                    ? state[((ulong)head * head_size + column) * head_size + d]
                    : 0.0f;
  }
  for (int t = 0; t < token_count; ++t) {
    float queries[4];
    float keys[4];
    for (int j = 0; j < 4; ++j) {
      const int d = j * 32 + lane;
      const ulong index = (ulong)t * channels + key_head * head_size + d;
      queries[j] = d < head_size ? ReadScalar(OMNI_SCALAR(qkv), index) : 0.0f;
      keys[j] = d < head_size ? ReadScalar(OMNI_SCALAR(qkv), index + key_size) : 0.0f;
    }
    const float inverse_size = native_recip((float)head_size);
    const float epsilon_scaled = epsilon * inverse_size;
    const float query_inverse =
        native_rsqrt(fma(DeltaNormSquareSum(queries, scratch), inverse_size, epsilon_scaled));
    const float key_inverse =
        native_rsqrt(fma(DeltaNormSquareSum(keys, scratch), inverse_size, epsilon_scaled));
    const float head_scale = 1.0f / sqrt((float)head_size);
    const ulong gate_index = (ulong)t * value_head_count + head;
    const float biased = ReadScalar(OMNI_SCALAR(alpha), gate_index) + time_bias[head];
    const float softplus = biased > 20.0f ? biased : native_log(1.0f + native_exp(biased));
    const float decay = native_exp(decay_weight[head] * softplus);
    const float update =
        native_recip(1.0f + native_exp(-ReadScalar(OMNI_SCALAR(beta), gate_index)));
    float projection = 0.0f;
    for (int j = 0; j < 4; ++j) {
      queries[j] = (queries[j] * query_inverse) * head_scale;
      keys[j] = (keys[j] * key_inverse) * head_scale;
      projection = fma(matrix[j], keys[j], projection);
    }
    projection = DeltaWarpSum(projection, scratch);
    const float value = is_valid ? ReadScalar(OMNI_SCALAR(qkv), (ulong)t * channels + 2 * key_size +
                                                                    head * head_size + column)
                                 : 0.0f;
    const float delta = fma(-decay, projection, value) * update;
    float result = 0.0f;
    for (int j = 0; j < 4; ++j) {
      matrix[j] = fma(decay, matrix[j], keys[j] * delta);
      result = fma(matrix[j], queries[j], result);
    }
    result = DeltaWarpSum(result, scratch);
    if (is_valid && lane == 0) {
      WriteScalar(OMNI_SCALAR(output), (ulong)t * value_size + head * head_size + column,
                  result * head_scale);
    }
  }
  for (int j = 0; j < 4; ++j) {
    const int d = j * 32 + lane;
    if (is_valid && d < head_size) {
      state[((ulong)head * head_size + column) * head_size + d] = matrix[j];
    }
  }
}

__kernel void SigmoidGate(__global uchar *input_buffer, ulong input_offset,
                          __global uchar *gate_buffer, ulong gate_offset, ulong element_count,
                          __global uchar *output_buffer, ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < element_count) {
    const float scale =
        native_recip(1.0f + native_exp(-ReadScalar(OMNI_SCALAR(gate), index)));
    WriteScalar(OMNI_SCALAR(output), index, ReadScalar(OMNI_SCALAR(input), index) * scale);
  }
}

inline void DeltaRotate(__global const scalar_t *input, __global const int *positions, ulong index,
                        int tokens, int heads, int head_dim, int rotary_dim, int temporal,
                        int height, int width, float theta_scale, __global scalar_t *output) {
  if (index >= (ulong)tokens * heads * head_dim) {
    return;
  }
  const int dimension = index % head_dim;
  if (dimension >= rotary_dim) {
    output[index] = input[index];
  } else if (dimension < rotary_dim / 2) {
    const int token = index / ((ulong)heads * head_dim);
    const int axis = dimension % 3 == 1 && dimension < 3 * height  ? 1
                     : dimension % 3 == 2 && dimension < 3 * width ? 2
                                                                   : 0;
    const float angle =
        (float)positions[(ulong)axis * tokens + token] * native_powr(theta_scale, (float)dimension);
    const float first = ReadScalar(input, index);
    const float second = ReadScalar(input, index + rotary_dim / 2);
    const float cosine = native_cos(angle);
    const float sine = native_sin(angle);
    WriteScalar(output, index, fma(first, cosine, -second * sine));
    WriteScalar(output, index + rotary_dim / 2, fma(first, sine, second * cosine));
  }
}

__kernel void PartialMultiAxisRope(__global uchar *query_buffer, ulong query_offset,
                                   __global uchar *key_buffer, ulong key_offset,
                                   __global uchar *position_ids_buffer, ulong position_ids_offset,
                                   int tokens, int query_heads, int key_value_heads, int head_dim,
                                   int rotary_dim, int temporal_section, int height_section,
                                   int width_section, float theta_scale,
                                   __global uchar *query_output_buffer, ulong query_output_offset,
                                   __global uchar *key_output_buffer, ulong key_output_offset) {
  const ulong index = get_global_id(0);
  DeltaRotate(OMNI_SCALAR(query), OMNI_INT(position_ids), index, tokens, query_heads, head_dim,
              rotary_dim, temporal_section, height_section, width_section, theta_scale,
              OMNI_SCALAR(query_output));
  DeltaRotate(OMNI_SCALAR(key), OMNI_INT(position_ids), index, tokens, key_value_heads, head_dim,
              rotary_dim, temporal_section, height_section, width_section, theta_scale,
              OMNI_SCALAR(key_output));
}
