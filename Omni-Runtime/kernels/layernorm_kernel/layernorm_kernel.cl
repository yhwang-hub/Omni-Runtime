// All three normalization entries use one 256-work-item group per row.
__kernel void RmsNorm(__global uchar *input_buffer, ulong input_offset,
                      __global uchar *weight_buffer, ulong weight_offset, int rows, int hidden_size,
                      float epsilon, __global uchar *output_buffer, ulong output_offset) {
  const ulong row = get_group_id(0);
  if (row >= (ulong)rows) {
    return;
  }
  const ulong first = row * hidden_size;
  float sum = 0.0f;
  for (int d = get_local_id(0); d < hidden_size; d += 256) {
    const float value = ReadScalar(OMNI_SCALAR(input), first + d);
    sum = fma(value, value, sum);
  }
  __local float scratch[256];
  sum = OmniBlockSum(sum, scratch);
  const float inverse = rsqrt(sum / (float)hidden_size + epsilon);
  for (int d = get_local_id(0); d < hidden_size; d += 256) {
    const float normalized = RoundScalar(ReadScalar(OMNI_SCALAR(input), first + d) * inverse);
    WriteScalar(OMNI_SCALAR(output), first + d, normalized * ReadScalar(OMNI_SCALAR(weight), d));
  }
}

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
  for (int d = get_local_id(0); d < hidden_size; d += 256) {
    const float value = ReadScalar(OMNI_SCALAR(input), first + d);
    sum += value;
    squares = fma(value, value, squares);
  }
  __local float scratch[256];
  sum = OmniBlockSum(sum, scratch);
  squares = OmniBlockSum(squares, scratch);
  const float mean = sum / (float)hidden_size;
  const float variance = fmax(squares / (float)hidden_size - mean * mean, 0.0f);
  const float inverse = rsqrt(variance + epsilon);
  for (int d = get_local_id(0); d < hidden_size; d += 256) {
    const float normalized = (ReadScalar(OMNI_SCALAR(input), first + d) - mean) * inverse;
    WriteScalar(
        OMNI_SCALAR(output), first + d,
        fma(normalized, ReadScalar(OMNI_SCALAR(weight), d), ReadScalar(OMNI_SCALAR(bias), d)));
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
  for (int d = get_local_id(0); d < hidden_size; d += 256) {
    const ulong index = first + d;
    const float value = RoundScalar(ReadScalar(OMNI_SCALAR(input), index) +
                                    ReadScalar(OMNI_SCALAR(residual), index));
    sum = fma(value, value, sum);
  }
  __local float scratch[256];
  sum = OmniBlockSum(sum, scratch);
  const float inverse = rsqrt(sum / (float)hidden_size + epsilon);
  for (int d = get_local_id(0); d < hidden_size; d += 256) {
    const ulong index = first + d;
    const float value = RoundScalar(ReadScalar(OMNI_SCALAR(input), index) +
                                    ReadScalar(OMNI_SCALAR(residual), index));
    WriteScalar(OMNI_SCALAR(residual_output), index, value);
    WriteScalar(OMNI_SCALAR(output), index,
                RoundScalar(value * inverse) * ReadScalar(OMNI_SCALAR(weight), d));
  }
}
