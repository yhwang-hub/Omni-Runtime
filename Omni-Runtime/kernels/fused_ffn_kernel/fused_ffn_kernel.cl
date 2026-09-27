inline float FfnGelu(float value) {
  return 0.5f * value *
         (1.0f + tanh(0.7978845608028654f * fma(0.044715f * value * value, value, value)));
}

// Eight logical CUDA warps per work-group; preserve the lane-strided FP32
// projection and shuffle-down summation order without requiring subgroups.
inline float FfnWarpSum(float value, __local float *scratch) {
  const uint id = get_local_id(0);
  const uint lane = id & 31U;
  scratch[id] = value;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint delta = 16U; delta != 0U; delta >>= 1U) {
    if (lane < delta) {
      scratch[id] += scratch[id + delta];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  value = scratch[id - lane];
  barrier(CLK_LOCAL_MEM_FENCE);
  return value;
}

__kernel void FusedFfn(__global uchar *input_buffer, ulong input_offset,
                       __global uchar *fc1_weight_buffer, ulong fc1_weight_offset,
                       __global uchar *fc1_bias_buffer, ulong fc1_bias_offset,
                       __global uchar *fc2_weight_buffer, ulong fc2_weight_offset,
                       __global uchar *fc2_bias_buffer, ulong fc2_bias_offset, int rows,
                       int input_size, int intermediate_size, int output_size,
                       __global uchar *output_buffer, ulong output_offset, __local float *cache) {
  const ulong row = get_group_id(0);
  if (row >= (ulong)rows) {
    return;
  }
  const int id = get_local_id(0);
  const int lane = id & 31;
  const int warp = id / 32;
  __local float scratch[256];
  __local float *intermediate = cache + input_size;
  for (int d = id; d < input_size; d += 256) {
    cache[d] = ReadScalar(OMNI_SCALAR(input), row * input_size + d);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (long first = 0; first < intermediate_size; first += 8) {
    const long neuron = first + warp;
    float value = 0.0f;
    if (neuron < intermediate_size) {
      for (int d = lane; d < input_size; d += 32) {
        value = fma(cache[d], ReadScalar(OMNI_SCALAR(fc1_weight), neuron * input_size + d), value);
      }
    }
    value = FfnWarpSum(value, scratch);
    if (lane == 0 && neuron < intermediate_size) {
      intermediate[neuron] = FfnGelu(value + ReadScalar(OMNI_SCALAR(fc1_bias), neuron));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (long first = 0; first < output_size; first += 8) {
    const long neuron = first + warp;
    float value = 0.0f;
    if (neuron < output_size) {
      for (int d = lane; d < intermediate_size; d += 32) {
        value = fma(intermediate[d],
                    ReadScalar(OMNI_SCALAR(fc2_weight), neuron * intermediate_size + d), value);
      }
    }
    value = FfnWarpSum(value, scratch);
    if (lane == 0 && neuron < output_size) {
      WriteScalar(OMNI_SCALAR(output), row * output_size + neuron,
                  value + ReadScalar(OMNI_SCALAR(fc2_bias), neuron));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
