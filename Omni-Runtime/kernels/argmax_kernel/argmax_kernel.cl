// One 256-work-item group per row; the sentinel and ties match CUDA.
__kernel void Argmax(__global uchar *logits_buffer, ulong logits_offset, int rows, int columns,
                     __global uchar *output_buffer, ulong output_offset) {
  const ulong row = get_group_id(0);
  if (row >= (ulong)rows) {
    return;
  }
  const uint id = get_local_id(0);
  float best_value = -FLT_MAX;
  int best_index = 0;
  for (int column = id; column < columns; column += 256) {
    const float value = ReadScalar(OMNI_SCALAR(logits), row * columns + column);
    if (value > best_value || (value == best_value && column < best_index)) {
      best_value = value;
      best_index = column;
    }
  }
  __local float values[256];
  __local int indices[256];
  values[id] = best_value;
  indices[id] = best_index;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint delta = 128; delta != 0; delta >>= 1) {
    if (id < delta) {
      const float other = values[id + delta];
      const int other_index = indices[id + delta];
      if (other > values[id] || (other == values[id] && other_index < indices[id])) {
        values[id] = other;
        indices[id] = other_index;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (id == 0) {
    OMNI_INT(output)[row] = indices[0];
  }
}
