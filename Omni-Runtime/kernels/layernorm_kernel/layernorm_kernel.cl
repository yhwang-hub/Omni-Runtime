#ifdef OMNI_GGML_NORMALIZATION
inline float GgmlRmsSquares(__global uchar *input_buffer, ulong input_offset,
                           __global uchar *residual_buffer, ulong residual_offset,
                           ulong first, int hidden_size, int has_residual,
                           __local float *partials) {
  const int lanes = hidden_size >= 1024 ? 1024 : 256;
  float sum = 0.0f;
  for (int logical = get_local_id(0); logical < lanes; logical += 256) {
    sum = 0.0f;
    for (int d = logical; d < hidden_size; d += lanes) {
      float value = ReadScalar(OMNI_SCALAR(input), first + d);
      if (has_residual) {
        value = RoundScalar(value + ReadScalar(OMNI_SCALAR(residual), first + d));
      }
      sum = fma(value, value, sum);
    }
    if (lanes == 1024) {
      partials[logical] = sum;
    }
  }
  if (lanes == 256) {
    return sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int distance = 16; distance > 0; distance /= 2) {
    for (int logical = get_local_id(0); logical < 1024; logical += 256) {
      if ((logical % 32) < distance) {
        partials[logical] += partials[logical + distance];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  return get_local_id(0) < 32 ? partials[get_local_id(0) * 32] : 0.0f;
}
#endif

__kernel void RmsNorm(__global uchar *input_buffer, ulong input_offset,
                      __global uchar *weight_buffer, ulong weight_offset, int rows, int hidden_size,
                      float epsilon, __global uchar *output_buffer, ulong output_offset) {
  const ulong row = get_group_id(0);
  if (row >= (ulong)rows) {
    return;
  }
  const ulong first = row * hidden_size;
  float sum = 0.0f;
#ifdef OMNI_GGML_NORMALIZATION
  __local float partials[1024];
  sum = GgmlRmsSquares(input_buffer, input_offset, input_buffer, input_offset,
                      first, hidden_size, 0, partials);
#else
  for (int d = get_local_id(0); d < hidden_size; d += 256) {
    const float value = ReadScalar(OMNI_SCALAR(input), first + d);
    sum = fma(value, value, sum);
  }
#endif
  __local float scratch[256];
  sum = OmniBlockSum(sum, scratch);
#ifdef OMNI_GGML_NORMALIZATION
  const float inverse = native_rsqrt(fma(sum, native_recip((float)hidden_size), epsilon));
#else
  const float inverse = rsqrt(sum / (float)hidden_size + epsilon);
#endif
  for (int d = get_local_id(0); d < hidden_size; d += 256) {
    const float normalized = RoundScalar(ReadScalar(OMNI_SCALAR(input), first + d) * inverse);
    WriteScalar(OMNI_SCALAR(output), first + d, normalized * ReadScalar(OMNI_SCALAR(weight), d));
  }
}

#ifndef OMNI_NORM_THREADS
#define OMNI_NORM_THREADS 256
#endif

__attribute__((reqd_work_group_size(OMNI_NORM_THREADS, 1, 1)))
__kernel void LayerNorm(__global uchar *input_buffer, ulong input_offset,
                        __global uchar *weight_buffer, ulong weight_offset,
                        __global uchar *bias_buffer, ulong bias_offset, int rows, int hidden_size,
                        float epsilon, __global uchar *output_buffer, ulong output_offset) {
  const ulong row = get_group_id(0);
  if (row >= (ulong)rows) {
    return;
  }
  const ulong first = row * hidden_size;
  float sum = 0.0f;
  float squares = 0.0f;
#ifdef OMNI_GGML_NORMALIZATION
  __local float2 partials[1024];
  if (hidden_size >= 1024) {
    // Preserve the reference's 1024 logical lanes with a portable 256-item workgroup.
    for (int logical = get_local_id(0); logical < 1024; logical += get_local_size(0)) {
      float2 partial = (float2)(0.0f);
      for (int d = logical; d < hidden_size; d += 1024) {
        const float value = ReadScalar(OMNI_SCALAR(input), first + d);
        partial.x += value;
        partial.y = fma(value, value, partial.y);
      }
      partials[logical] = partial;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int distance = 16; distance > 0; distance /= 2) {
      for (int logical = get_local_id(0); logical < 1024; logical += get_local_size(0)) {
        if ((logical % 32) < distance) {
          partials[logical] += partials[logical + distance];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = get_local_id(0) < 32 ? partials[get_local_id(0) * 32].x : 0.0f;
    squares = get_local_id(0) < 32 ? partials[get_local_id(0) * 32].y : 0.0f;
  } else
#endif
  {
  for (int d = get_local_id(0); d < hidden_size; d += get_local_size(0)) {
    const float value = ReadScalar(OMNI_SCALAR(input), first + d);
    sum += value;
    squares = fma(value, value, squares);
  }
  }
  __local float scratch[256];
  sum = OmniBlockSum(sum, scratch);
  squares = OmniBlockSum(squares, scratch);
#ifdef OMNI_GGML_NORMALIZATION
  const float inverse_size = native_recip((float)hidden_size);
  const float mean = sum * inverse_size;
  const float variance = fma(squares, inverse_size, -(mean * mean));
  const float inverse = native_rsqrt(variance + epsilon);
#else
  const float mean = sum / (float)hidden_size;
  const float variance = fmax(squares / (float)hidden_size - mean * mean, 0.0f);
  const float inverse = rsqrt(variance + epsilon);
#endif
  for (int d = get_local_id(0); d < hidden_size; d += get_local_size(0)) {
    const float normalized = (ReadScalar(OMNI_SCALAR(input), first + d) - mean) * inverse;
#ifdef OMNI_GGML_NORMALIZATION
    const float weighted = normalized * ReadScalar(OMNI_SCALAR(weight), d);
    WriteScalar(OMNI_SCALAR(output), first + d, weighted + ReadScalar(OMNI_SCALAR(bias), d));
#else
    WriteScalar(
        OMNI_SCALAR(output), first + d,
        fma(normalized, ReadScalar(OMNI_SCALAR(weight), d), ReadScalar(OMNI_SCALAR(bias), d)));
#endif
  }
}

__kernel void ResidualRmsNorm(__global uchar *input_buffer, ulong input_offset,
                              __global uchar *residual_buffer, ulong residual_offset,
                              __global uchar *weight_buffer, ulong weight_offset, int rows,
                              int hidden_size, float epsilon,
                              __global uchar *residual_output_buffer, ulong residual_output_offset,
                              __global uchar *output_buffer, ulong output_offset) {
  const ulong row = get_group_id(0);
  if (row >= (ulong)rows) {
    return;
  }
  const ulong first = row * hidden_size;
  float sum = 0.0f;
#ifdef OMNI_GGML_NORMALIZATION
  __local float partials[1024];
  sum = GgmlRmsSquares(input_buffer, input_offset, residual_buffer, residual_offset,
                      first, hidden_size, 1, partials);
#else
  for (int d = get_local_id(0); d < hidden_size; d += 256) {
    const ulong index = first + d;
    const float value = RoundScalar(ReadScalar(OMNI_SCALAR(input), index) +
                                    ReadScalar(OMNI_SCALAR(residual), index));
    sum = fma(value, value, sum);
  }
#endif
  __local float scratch[256];
  sum = OmniBlockSum(sum, scratch);
#ifdef OMNI_GGML_NORMALIZATION
  const float inverse = native_rsqrt(fma(sum, native_recip((float)hidden_size), epsilon));
#else
  const float inverse = rsqrt(sum / (float)hidden_size + epsilon);
#endif
  for (int d = get_local_id(0); d < hidden_size; d += 256) {
    const ulong index = first + d;
    const float value = RoundScalar(ReadScalar(OMNI_SCALAR(input), index) +
                                    ReadScalar(OMNI_SCALAR(residual), index));
    WriteScalar(OMNI_SCALAR(residual_output), index, value);
    WriteScalar(OMNI_SCALAR(output), index,
                RoundScalar(value * inverse) * ReadScalar(OMNI_SCALAR(weight), d));
  }
}
