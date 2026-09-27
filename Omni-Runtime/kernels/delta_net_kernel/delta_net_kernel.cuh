#pragma once

#include <kernels/common/kernel_type_utils.h>
#include <kernels/mrope_kernel/mrope_kernel.cuh>

namespace omni_runtime::kernels::delta_net_kernel {

template <typename T>
__global__ void CausalConvImpl(const T *const input, const float *const weight, float *const state,
                               const int32_t tokens, const int32_t channels,
                               const int32_t kernel_size, T *const output) {
  const int32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) {
    return;
  }
  float history[15] = {};
  for (int32_t j = 0; j < kernel_size - 1; ++j) {
    history[j] = state[static_cast<uint64_t>(channel) * (kernel_size - 1) + j];
  }
  for (int32_t t = 0; t < tokens; ++t) {
    const uint64_t index = static_cast<uint64_t>(t) * channels + channel;
    const float current = ToFloat(input[index]);
    float sum = 0.0F;
    for (int32_t j = 0; j < kernel_size - 1; ++j) {
      sum = fmaf(history[j], weight[channel * kernel_size + j], sum);
    }
    sum = fmaf(current, weight[channel * kernel_size + kernel_size - 1], sum);
    output[index] = FromFloat<T>(__fdividef(sum, 1.0F + __expf(-sum)));
    for (int32_t j = 0; j < kernel_size - 2; ++j) {
      history[j] = history[j + 1];
    }
    if (kernel_size > 1) {
      history[kernel_size - 2] = current;
    }
  }
  for (int32_t j = 0; j < kernel_size - 1; ++j) {
    state[static_cast<uint64_t>(channel) * (kernel_size - 1) + j] = history[j];
  }
}

__device__ inline float DeltaWarpSum(float value) {
#pragma unroll
  for (int32_t distance = 16; distance > 0; distance >>= 1) {
    value += __shfl_xor_sync(0xFFFFFFFFU, value, distance);
  }
  return value;
}

__device__ inline float NormSquareSum(const float *const values) {
  float warps[4];
#pragma unroll
  for (int32_t part = 0; part < 4; ++part) {
    warps[part] = DeltaWarpSum(__fmul_rn(values[part], values[part]));
  }
  return (warps[0] + warps[2]) + (warps[1] + warps[3]);
}

template <typename T, int32_t kColumns>
__global__ void DeltaNetImpl(const T *const qkv, const T *const alpha, const T *const beta,
                             const float *const decay_weight, const float *const time_bias,
                             float *const state, const int32_t tokens, const int32_t key_heads,
                             const int32_t value_heads, const int32_t head_size,
                             const float epsilon, T *const output) {
  const int32_t head = blockIdx.x;
  const int32_t first_column = (blockIdx.y * 4 + threadIdx.x / 32) * kColumns;
  const int32_t lane = threadIdx.x % 32;
  const int32_t key_head = head % key_heads;
  const int32_t key_size = key_heads * head_size;
  const int32_t value_size = value_heads * head_size;
  const int32_t channels = 2 * key_size + value_size;
  float matrix[kColumns][4] = {};
#pragma unroll
  for (int32_t local = 0; local < kColumns; ++local) {
    const int32_t column = first_column + local;
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
      const int32_t d = j * 32 + lane;
      if (column < head_size && d < head_size) {
        matrix[local][j] =
            state[(static_cast<uint64_t>(head) * head_size + column) * head_size + d];
      }
    }
  }
  for (int32_t t = 0; t < tokens; ++t) {
    float queries[4] = {};
    float keys[4] = {};
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
      const int32_t d = j * 32 + lane;
      if (d < head_size) {
        const uint64_t index = static_cast<uint64_t>(t) * channels + key_head * head_size + d;
        queries[j] = ToFloat(qkv[index]);
        keys[j] = ToFloat(qkv[index + key_size]);
      }
    }
    const float inverse_size = __fdividef(1.0F, static_cast<float>(head_size));
    const float epsilon_scaled = epsilon * inverse_size;
    const float query_inverse = rsqrtf(fmaf(NormSquareSum(queries), inverse_size, epsilon_scaled));
    const float key_inverse = rsqrtf(fmaf(NormSquareSum(keys), inverse_size, epsilon_scaled));
    const float head_scale = __fdiv_rn(1.0F, __fsqrt_rn(static_cast<float>(head_size)));
    const uint64_t gate_index = static_cast<uint64_t>(t) * value_heads + head;
    const float biased = ToFloat(alpha[gate_index]) + time_bias[head];
    const float softplus = biased > 20.0F ? biased : __logf(1.0F + __expf(biased));
    const float decay = __expf(decay_weight[head] * softplus);
    const float update = __fdividef(1.0F, 1.0F + __expf(-ToFloat(beta[gate_index])));
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
      queries[j] = __fmul_rn(__fmul_rn(queries[j], query_inverse), head_scale);
      keys[j] = __fmul_rn(__fmul_rn(keys[j], key_inverse), head_scale);
    }
#pragma unroll
    for (int32_t local = 0; local < kColumns; ++local) {
      const int32_t column = first_column + local;
      const bool is_valid = column < head_size;
      float projection = 0.0F;
#pragma unroll
      for (int32_t j = 0; j < 4; ++j) {
        projection = fmaf(matrix[local][j], keys[j], projection);
      }
      projection = DeltaWarpSum(projection);
      const float value = is_valid ? ToFloat(qkv[static_cast<uint64_t>(t) * channels +
                                                 2 * key_size + head * head_size + column])
                                   : 0.0F;
      const float delta = fmaf(-decay, projection, value) * update;
      float result = 0.0F;
#pragma unroll
      for (int32_t j = 0; j < 4; ++j) {
        matrix[local][j] = fmaf(decay, matrix[local][j], keys[j] * delta);
        result = fmaf(matrix[local][j], queries[j], result);
      }
      result = DeltaWarpSum(result);
      if (is_valid && lane == 0) {
        output[static_cast<uint64_t>(t) * value_size + head * head_size + column] =
            FromFloat<T>(result * head_scale);
      }
    }
  }
#pragma unroll
  for (int32_t local = 0; local < kColumns; ++local) {
    const int32_t column = first_column + local;
#pragma unroll
    for (int32_t j = 0; j < 4; ++j) {
      const int32_t d = j * 32 + lane;
      if (column < head_size && d < head_size) {
        state[(static_cast<uint64_t>(head) * head_size + column) * head_size + d] =
            matrix[local][j];
      }
    }
  }
}

template <typename T>
__global__ void SigmoidGateImpl(const T *const input, const T *const gate, const uint64_t count,
                                T *const output) {
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    const float scale = __fdividef(1.0F, 1.0F + __expf(-ToFloat(gate[index])));
    output[index] = FromFloat<T>(ToFloat(input[index]) * scale);
  }
}

template <typename T>
__device__ void RotatePartial(const T *const input, const int32_t *const positions,
                              const uint64_t index, const int32_t tokens, const int32_t heads,
                              const int32_t head_dim, const int32_t rotary_dim,
                              const int32_t temporal, const int32_t height, const int32_t width,
                              const float theta_scale, T *const output) {
  if (index >= static_cast<uint64_t>(tokens) * heads * head_dim) {
    return;
  }
  const int32_t dimension = index % head_dim;
  if (dimension >= rotary_dim) {
    output[index] = input[index];
  } else if (dimension < rotary_dim / 2) {
    const int32_t token = index / (static_cast<uint64_t>(heads) * head_dim);
    const int32_t axis = mrope_kernel::PositionAxis(dimension, temporal, height, width);
    const float angle =
        static_cast<float>(positions[static_cast<uint64_t>(axis) * tokens + token]) *
        __powf(theta_scale, static_cast<float>(dimension));
    const float first = ToFloat(input[index]);
    const float second = ToFloat(input[index + rotary_dim / 2]);
    const float cosine = __cosf(angle);
    const float sine = __sinf(angle);
    output[index] = FromFloat<T>(fmaf(first, cosine, -second * sine));
    output[index + rotary_dim / 2] = FromFloat<T>(fmaf(first, sine, second * cosine));
  }
}

template <typename T>
__global__ void
PartialRopeImpl(const T *const query, const T *const key, const int32_t *const positions,
                const int32_t tokens, const int32_t query_heads, const int32_t key_heads,
                const int32_t head_dim, const int32_t rotary_dim, const int32_t temporal,
                const int32_t height, const int32_t width, const float theta_scale,
                T *const query_output, T *const key_output) {
  const uint64_t count = static_cast<uint64_t>(tokens) * max(query_heads, key_heads) * head_dim;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    RotatePartial(query, positions, index, tokens, query_heads, head_dim, rotary_dim, temporal,
                  height, width, theta_scale, query_output);
    RotatePartial(key, positions, index, tokens, key_heads, head_dim, rotary_dim, temporal, height,
                  width, theta_scale, key_output);
  }
}

} // namespace omni_runtime::kernels::delta_net_kernel
