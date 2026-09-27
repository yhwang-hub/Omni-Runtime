__kernel void FlowStep(__global uchar *state_buffer, ulong state_offset,
                       __global uchar *velocity_buffer, ulong velocity_offset, float dt,
                       ulong element_count, __global uchar *output_buffer, ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < element_count) {
    WriteScalar(
        index,
        fma(dt, ReadScalar(OMNI_SCALAR(velocity), index), ReadScalar(OMNI_SCALAR(state), index)),
        OMNI_SCALAR(output));
  }
}
