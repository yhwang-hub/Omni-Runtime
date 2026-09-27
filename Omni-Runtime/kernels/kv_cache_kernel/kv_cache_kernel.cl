__kernel void StoreKvCache(__global uchar *key_buffer, ulong key_offset,
                           __global uchar *value_buffer, ulong value_offset, int token_count,
                           int key_value_size, int start_position, int cache_capacity,
                           __global uchar *key_cache_buffer, ulong key_cache_offset,
                           __global uchar *value_cache_buffer, ulong value_cache_offset,
                           __global uchar *cache_position_buffer, ulong cache_position_offset,
                           int has_cache_position) {
  const ulong index = get_global_id(0);
  if (index < (ulong)token_count * key_value_size) {
    const int start = has_cache_position ? OMNI_INT(cache_position)[0] : start_position;
    const long token = (long)start + (long)(index / key_value_size);
    if (token >= 0 && token < cache_capacity) {
      const ulong destination = (ulong)token * key_value_size + index % key_value_size;
      OMNI_SCALAR(key_cache)[destination] = OMNI_SCALAR(key)[index];
      OMNI_SCALAR(value_cache)[destination] = OMNI_SCALAR(value)[index];
    }
  }
}

__kernel void AdvanceSequence(__global uchar *positions_buffer, ulong positions_offset,
                              __global uchar *cache_position_buffer, ulong cache_position_offset) {
  if (get_global_id(0) == 0) {
    ++OMNI_INT(positions)[0];
    ++OMNI_INT(positions)[1];
    ++OMNI_INT(positions)[2];
    ++OMNI_INT(cache_position)[0];
  }
}
