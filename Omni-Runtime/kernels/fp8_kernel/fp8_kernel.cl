#pragma OPENCL FP_CONTRACT OFF

inline float Fp8Decode(const uchar code) {
  const uint magnitude = code & 127;
  const uint exponent = magnitude >> 3;
  const uint mantissa = magnitude & 7;
  const float value = exponent == 0 ? (float)mantissa * 0.001953125f
                                   : as_float(((exponent + 120) << 23) | (mantissa << 20));
  return magnitude == 127 ? NAN : (code & 128) ? -value : value;
}

inline float Fp8Quantize(const float value) {
  if (isnan(value)) {
    return NAN;
  }
  const float magnitude = fmin(fabs(value), 448.0f);
  const uint bits = as_uint(magnitude);
  const int exponent = (int)((bits >> 23) & 255) - 127;
  const int center = clamp((exponent + 7) * 8 + (int)((bits >> 20) & 7), 0, 126);
  int best = 0;
  float error = magnitude;
  for (int code = max(0, center - 8); code <= min(126, center + 8); ++code) {
    const float distance = fabs(magnitude - Fp8Decode((uchar)code));
    if (distance < error || (distance == error && code % 2 == 0)) {
      error = distance;
      best = code;
    }
  }
  return copysign(Fp8Decode((uchar)best), value);
}

__kernel void Fp8Linear(__global uchar *input_buffer, ulong input_offset,
                        __global uchar *weight_buffer, ulong weight_offset,
                        __global uchar *weight_scales_buffer, ulong weight_scales_offset,
                        int rows, int inner, int columns, int block_reduction_groups,
                        __global uchar *output_buffer, ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index >= (ulong)rows * columns) {
    return;
  }
  const ulong row = index / columns;
  const int column = index % columns;
  __global const uchar *weight = weight_buffer + weight_offset;
  __global const float *scales = (__global const float *)(weight_scales_buffer + weight_scales_offset);
  float result = 0.0f;
  const int blocks = inner / 128;
  for (int block = 0; block < blocks; ++block) {
    const int first = (block % block_reduction_groups * (blocks / block_reduction_groups) +
                       block / block_reduction_groups) * 128;
    float maximum = 1e-12f;
    for (int d = 0; d < 128; ++d) {
      maximum = fmax(maximum, fabs(ReadScalar(OMNI_SCALAR(input), row * inner + first + d)));
    }
    const float activation_scale = maximum * (1.0f / 448.0f);
    long dot = 0L;
    bool has_nan = false;
    for (int d = 0; d < 128; ++d) {
      const float value = ReadScalar(OMNI_SCALAR(input), row * inner + first + d);
      const float activation = Fp8Quantize(value / activation_scale);
      const float decoded_weight = Fp8Decode(weight[(ulong)column * inner + first + d]);
      has_nan = has_nan || !isfinite(activation) || !isfinite(decoded_weight);
      if (isfinite(activation) && isfinite(decoded_weight)) {
        dot += (long)convert_int_rte(activation * 512.0f) *
               convert_int_rte(decoded_weight * 512.0f);
      }
    }
    const float partial = has_nan ? NAN : convert_float_rte(dot) * (1.0f / 262144.0f);
    result += (partial * activation_scale) * scales[(ulong)(column / 128) * (inner / 128) + first / 128];
  }
  WriteScalar(index, result, OMNI_SCALAR(output));
}
