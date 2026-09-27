#ifndef OMNI_OPENCL_DECODE_GROUPED
#define OMNI_OPENCL_DECODE_GROUPED

#if defined(OMNI_NVIDIA_MMA) && defined(OMNI_NVIDIA_WARP) && (OMNI_DTYPE == 1 || OMNI_DTYPE == 2)
// Same Attention ABI. Local (128,1), global (kv_heads*128,ceil(cache_capacity/128)).
// Partials are FP32 [partition,query_head,head_dim+2]: weighted values, max, sum.
__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void AttentionDecodeGrouped(
    __global uchar *query_buffer, ulong query_offset, __global uchar *key_cache_buffer,
    ulong key_cache_offset, __global uchar *value_cache_buffer, ulong value_cache_offset,
    int query_token_count, int query_head_count, int key_value_head_count, int cached_token_count,
    int start_position, int head_dim, int is_causal, __global uchar *output_buffer,
    ulong output_offset, __global uchar *decode_partials_buffer, ulong decode_partials_offset,
    __global uchar *cache_position_buffer, ulong cache_position_offset, int has_decode_partials,
    int has_cache_position) {
  const int kv_head = get_group_id(0);
  if (query_token_count != 1 || head_dim != 128 || !has_decode_partials ||
      query_head_count != key_value_head_count * 4 || kv_head >= key_value_head_count) {
    return;
  }
  const int first_token = get_group_id(1) * 128;
  const int position = has_cache_position ? OMNI_INT(cache_position)[0] : start_position;
  const int cached_tokens = has_cache_position ? position + 1 : cached_token_count;
  const int token_count = is_causal ? min(cached_tokens, position + 1) : cached_tokens;
  if (first_token >= token_count) {
    return;
  }
  const int count = min(128, token_count - first_token);
  const int id = get_local_id(0);
  const int lane = id % 32;
  const int warp = id / 32;
  const int row = lane / 4;
  const int column = (lane % 4) * 2;
  __local ushort cache[128 * 136] __attribute__((aligned(16)));
  __local float scores[4][128];
  for (int index = id * 8; index < 128 * 128; index += 128 * 8) {
    const int token = index / 128;
    const int d = index % 128;
    uint4 packed = (uint4)(0);
    if (token < count) {
      const __global scalar_t *source =
          OMNI_SCALAR(key_cache) +
          ((ulong)(first_token + token) * key_value_head_count + kv_head) * 128 + d;
      packed = (key_cache_offset & 15UL) == 0 ? *(__global const uint4 *)source
                                              : as_uint4(vload8(0, source));
    }
    *(__local uint4 *)(cache + token * 136 + d) = packed;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float fragments[4][4] = {{0.0f}};
#pragma unroll
  for (int d = 0; d < 128; d += 16) {
    const int offset = (kv_head * 4 + row) * 128 + d + column;
    const uint a[4] = {
        row < 4 ? OmniPack(OMNI_SCALAR(query)[offset], OMNI_SCALAR(query)[offset + 1]) : 0U, 0U,
        row < 4 ? OmniPack(OMNI_SCALAR(query)[offset + 8], OMNI_SCALAR(query)[offset + 9]) : 0U,
        0U};
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
      const int index = (warp * 32 + tile * 8 + row) * 136 + d + column;
      const uint b[2] = {OmniLocalPair(cache, index), OmniLocalPair(cache, index + 8)};
      OmniMma(a, b, fragments[tile]);
    }
  }
  if (row < 4) {
#pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
#pragma unroll
      for (int item = 0; item < 2; ++item) {
        const int token = warp * 32 + tile * 8 + column + item;
        scores[row][token] = token < count ? fragments[tile][item] * rsqrt(128.0f) : -INFINITY;
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float maximum = -INFINITY;
#pragma unroll
  for (int token = lane; token < 128; token += 32) {
    maximum = fmax(maximum, scores[warp][token]);
  }
  for (uint delta = 16; delta != 0; delta >>= 1) {
    float other;
    asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=f"(other) : "f"(maximum), "r"(delta));
    maximum = fmax(maximum, other);
  }
  float broadcast;
  asm("shfl.sync.idx.b32 %0, %1, 0, 31, -1;" : "=f"(broadcast) : "f"(maximum));
  maximum = broadcast;
  float denominator = 0.0f;
#pragma unroll
  for (int token = lane; token < 128; token += 32) {
    const float probability = exp(scores[warp][token] - maximum);
    scores[warp][token] = probability;
    denominator += probability;
  }
  denominator = OmniWarpSum(denominator);
  for (int index = id * 8; index < 128 * 128; index += 128 * 8) {
    const int token = index / 128;
    const int d = index % 128;
    if (token < count) {
      const __global scalar_t *source =
          OMNI_SCALAR(value_cache) +
          ((ulong)(first_token + token) * key_value_head_count + kv_head) * 128 + d;
      const uint4 packed = (value_cache_offset & 15UL) == 0 ? *(__global const uint4 *)source
                                                            : as_uint4(vload8(0, source));
      *(__local uint4 *)(cache + token * 136 + d) = packed;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float accumulators[4] = {0.0f};
  for (int token = 0; token < count; ++token) {
    const float probability = scores[warp][token];
    const uint2 packed = *(__local const uint2 *)(cache + token * 136 + lane * 4);
    accumulators[0] = fma(probability, OmniPackedScalar(packed.s0), accumulators[0]);
    accumulators[1] = fma(probability, OmniPackedScalar(packed.s0 >> 16), accumulators[1]);
    accumulators[2] = fma(probability, OmniPackedScalar(packed.s1), accumulators[2]);
    accumulators[3] = fma(probability, OmniPackedScalar(packed.s1 >> 16), accumulators[3]);
  }
  const int head = kv_head * 4 + warp;
  const ulong output_index = ((ulong)get_group_id(1) * query_head_count + head) * 130;
  __global float *partials = (__global float *)(decode_partials_buffer + decode_partials_offset);
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    partials[output_index + lane * 4 + item] = accumulators[item];
  }
  if (lane == 0) {
    partials[output_index + 128] = maximum;
    partials[output_index + 129] = denominator;
  }
}
#endif

#endif
