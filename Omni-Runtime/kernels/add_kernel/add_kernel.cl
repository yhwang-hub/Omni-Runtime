__kernel void Add(__global uchar *lhs_buffer, ulong lhs_offset, __global uchar *rhs_buffer,
                  ulong rhs_offset, ulong element_count, __global uchar *output_buffer,
                  ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < element_count) {
    WriteScalar(OMNI_SCALAR(output), index,
                ReadScalar(OMNI_SCALAR(lhs), index) + ReadScalar(OMNI_SCALAR(rhs), index));
  }
}
