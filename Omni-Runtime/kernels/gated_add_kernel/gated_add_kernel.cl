__kernel void GatedAdd(__global uchar *lhs_buffer, ulong lhs_offset, __global uchar *rhs_buffer,
                       ulong rhs_offset, __global uchar *gate_buffer, ulong gate_offset,
                       ulong element_count, int hidden_size, __global uchar *output_buffer,
                       ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < element_count) {
    WriteScalar(OMNI_SCALAR(output), index,
                fma(ReadScalar(OMNI_SCALAR(rhs), index),
                    ReadScalar(OMNI_SCALAR(gate), index % hidden_size),
                    ReadScalar(OMNI_SCALAR(lhs), index)));
  }
}
