__kernel void SplitGatedQuery(__global uchar *query_gate_buffer, ulong query_gate_offset,
                              int token_count, int head_count, int head_size,
                              __global uchar *query_buffer, ulong query_offset,
                              __global uchar *gate_buffer, ulong gate_offset) {
  const ulong index = get_global_id(0);
  const ulong count = (ulong)token_count * head_count * head_size;
  if (index < count) {
    const ulong source = (index / head_size) * (2UL * head_size) + index % head_size;
    // This is a layout operation; preserve scalar bits, including NaN payloads.
    OMNI_SCALAR(query)[index] = OMNI_SCALAR(query_gate)[source];
    OMNI_SCALAR(gate)[index] = OMNI_SCALAR(query_gate)[source + head_size];
  }
}
