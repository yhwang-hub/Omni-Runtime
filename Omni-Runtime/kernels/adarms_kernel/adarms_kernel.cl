__kernel void AdaRmsNorm(__global uchar *input_buffer, ulong input_offset,
                         __global uchar *scale_buffer, ulong scale_offset,
                         __global uchar *shift_buffer, ulong shift_offset, int rows,
                         int hidden_size, float epsilon, __global uchar *output_buffer,
                         ulong output_offset) {
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
    // AdaRMS keeps normalization and modulation in FP32 until the final store.
    const float normalized = ReadScalar(OMNI_SCALAR(input), first + d) * inverse;
    WriteScalar(OMNI_SCALAR(output), first + d,
                fma(normalized, 1.0f + ReadScalar(OMNI_SCALAR(scale), d),
                    ReadScalar(OMNI_SCALAR(shift), d)));
  }
}
