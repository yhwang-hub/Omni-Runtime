__kernel void SwiGLU(__global uchar *gate_buffer, ulong gate_offset, __global uchar *up_buffer,
                     ulong up_offset, ulong element_count, __global uchar *output_buffer,
                     ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < element_count) {
    const float value = ReadScalar(OMNI_SCALAR(gate), index);
#ifdef OMNI_GGML_SILU
    const float silu = RoundScalar(native_divide(value, 1.0f + native_exp(-value)));
#else
    const float silu = RoundScalar(value / (1.0f + exp(-value)));
#endif
    WriteScalar(OMNI_SCALAR(output), index, silu * ReadScalar(OMNI_SCALAR(up), index));
  }
}
