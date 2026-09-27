// Flat token_count * hidden_size work-items, rounded to a local size of 256.
__kernel void Embedding(__global uchar *indices_buffer, ulong indices_offset,
                        __global uchar *weight_buffer, ulong weight_offset, int token_count,
                        int hidden_size, __global uchar *output_buffer, ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < (ulong)token_count * hidden_size) {
    const ulong token = index / hidden_size;
    const ulong source = (ulong)OMNI_INT(indices)[token] * hidden_size + index % hidden_size;
    OMNI_SCALAR(output)[index] = OMNI_SCALAR(weight)[source];
  }
}
