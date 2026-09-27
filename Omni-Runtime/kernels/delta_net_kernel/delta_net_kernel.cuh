#pragma once

#include <type_traits>

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
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      sum = ToFloat(FromFloat<T>(sum));
    }
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      output[index] = FromFloat<T>(sum / (1.0F + expf(-sum)));
    } else {
      output[index] = FromFloat<T>(__fdividef(sum, 1.0F + __expf(-sum)));
    }
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

template <typename T>
__global__ void PrepareChunkImpl(const T *const qkv,
                                 const T *const alpha,
                                 const T *const beta,
                                 const float *const decay_weight, const float *const time_bias,
                                 const int32_t tokens, const int32_t key_heads,
                                 const int32_t value_heads, const int32_t size,
                                 const float epsilon, const bool is_log_decay,
                                 float *const workspace) {
  const int32_t head = blockIdx.x;
  const int32_t token = blockIdx.y;
  const int32_t lane = threadIdx.x;
  const int32_t padded = (tokens + 63) / 64 * 64;
  const uint64_t values = static_cast<uint64_t>(value_heads) * padded * size;
  const uint64_t matrices = static_cast<uint64_t>(value_heads) * padded * 64;
  const uint64_t offset = (static_cast<uint64_t>(head) * padded + token) * size;
  const int32_t key_size = key_heads * size;
  const int32_t channels = (2 * key_heads + value_heads) * size;
  float queries[4] = {};
  float keys[4] = {};
  float query_square = 0.0F;
  float key_square = 0.0F;
  for (int32_t part = 0; part < 4; ++part) {
    const int32_t d = lane * 4 + part;
    if (token < tokens && d < size) {
      const uint64_t source = static_cast<uint64_t>(token) * channels + head % key_heads * size + d;
      queries[part] = ToFloat(qkv[source]);
      keys[part] = ToFloat(qkv[source + key_size]);
    }
    query_square += __fmul_rn(queries[part], queries[part]);
    key_square += __fmul_rn(keys[part], keys[part]);
  }
  query_square = __shfl_sync(0xFFFFFFFFU, WarpReduceSum(query_square), 0);
  key_square = __shfl_sync(0xFFFFFFFFU, WarpReduceSum(key_square), 0);
  const float query_inverse = rsqrtf(query_square + epsilon);
  const float key_inverse = rsqrtf(key_square + epsilon);
  const uint64_t gate = static_cast<uint64_t>(token) * value_heads + head;
  const float update = token < tokens
                           ? ToFloat(FromFloat<T>(
                                 1.0F / (1.0F + expf(-ToFloat(beta[gate])))))
                           : 0.0F;
  for (int32_t part = 0; part < 4; ++part) {
    const int32_t d = lane * 4 + part;
    if (d < size) {
      const float q = __fmul_rn(queries[part], query_inverse);
      const float k = __fmul_rn(keys[part], key_inverse);
      const float v = token < tokens
                          ? ToFloat(qkv[static_cast<uint64_t>(token) * channels + 2 * key_size +
                                        head * size + d])
                          : 0.0F;
      workspace[offset + d] = q * static_cast<float>(1.0 / sqrt(128.0));
      workspace[values + offset + d] = k;
      workspace[2 * values + offset + d] = k * update;
      workspace[3 * values + offset + d] = v * update;
    }
  }
  if (lane == 0) {
    const float biased = token < tokens ? ToFloat(alpha[gate]) + time_bias[head] : 0.0F;
    const float softplus = biased > 20.0F ? biased : log1pf(expf(biased));
    const float decay = is_log_decay ? -expf(decay_weight[head]) : decay_weight[head];
    workspace[7 * values + 2 * matrices + static_cast<uint64_t>(head) * padded + token] =
        token < tokens ? decay * softplus : 0.0F;
  }
}

template <typename T>
__global__ void TransformChunkImpl(const int32_t padded, const int32_t heads,
                                   const int32_t size, float *const workspace) {
  const int32_t batch = blockIdx.x;
  const uint64_t values = static_cast<uint64_t>(heads) * padded * size;
  const uint64_t matrices = static_cast<uint64_t>(heads) * padded * 64;
  float *const gates = workspace + 7 * values + 2 * matrices + batch * 64;
  __shared__ float cumulative[64];
  if (threadIdx.x < 64) {
    cumulative[threadIdx.x] = gates[threadIdx.x];
  }
  __syncthreads();
  // Larger batches carry the first 32-value scan into the second segment.
  const int32_t scan_size = heads * (padded / 64) > 32 ? 32 : 64;
  for (int32_t first = 0; first < 64; first += scan_size) {
    if (threadIdx.x == first) {
      cumulative[first] += first == 0 ? 0.0F : cumulative[first - 1];
    }
    __syncthreads();
    for (int32_t distance = 1; distance < scan_size; distance <<= 1) {
      if (threadIdx.x >= first && threadIdx.x < first + scan_size &&
          (threadIdx.x & distance) != 0) {
        const int32_t source = (threadIdx.x & ~(2 * distance - 1)) + distance - 1;
        cumulative[threadIdx.x] += cumulative[source];
      }
      __syncthreads();
    }
  }
  for (int32_t index = threadIdx.x; index < 64 * 64; index += blockDim.x) {
    const int32_t row = index / 64;
    const int32_t column = index % 64;
    const float decay = row >= column ? expf(cumulative[row] - cumulative[column]) : 0.0F;
    const uint64_t offset = static_cast<uint64_t>(batch) * 4096 + index;
    workspace[7 * values + offset] *= decay;
    workspace[7 * values + matrices + offset] *= decay;
  }
  for (int32_t index = threadIdx.x; index < 64 * size; index += blockDim.x) {
    const int32_t token = index / size;
    const uint64_t offset = static_cast<uint64_t>(batch) * 64 * size + index;
    workspace[offset] *= expf(cumulative[token]);
    workspace[values + offset] *= expf(cumulative[63] - cumulative[token]);
    workspace[2 * values + offset] *= expf(cumulative[token]);
  }
  if (threadIdx.x == 0) {
    gates[0] = expf(cumulative[63]);
  }
}

template <typename T>
__global__ void SolveChunkImpl(const int32_t padded, const int32_t heads,
                               const int32_t size, float *const workspace) {
  const int32_t batches = heads * (padded / 64);
  const int32_t batch = blockIdx.x;
  const int32_t column = threadIdx.x;
  const uint64_t values = static_cast<uint64_t>(heads) * padded * size;
  const float *const system = workspace + 7 * values + static_cast<uint64_t>(batch) * 4096;
  float *const rhs = workspace + (blockIdx.y == 0 ? 2 : 3) * values +
                     static_cast<uint64_t>(batch) * 64 * size;
  __shared__ float solved[64 * 128];
  for (int32_t row = 0; row < 64; ++row) {
    float value = rhs[row * size + column];
    if (batches <= 8) {
      float sum = 0.0F;
      for (int32_t previous = 0; previous < row; ++previous) {
        sum = fmaf(system[row * 64 + previous], solved[previous * size + column], sum);
      }
      value -= sum;
    } else {
      // The reference batched solver rounds each completed eight-row update separately.
      const int32_t first = row / 8 * 8;
      for (int32_t block = 0; block < first; block += 8) {
        float sum = 0.0F;
        for (int32_t previous = block; previous < block + 8; ++previous) {
          sum = fmaf(system[row * 64 + previous], solved[previous * size + column], sum);
        }
        value -= sum;
      }
      for (int32_t previous = first; previous < row; ++previous) {
        value = fmaf(-system[row * 64 + previous], solved[previous * size + column], value);
      }
    }
    solved[row * size + column] = value;
    rhs[row * size + column] = value;
  }
}

template <typename T>
__global__ void ChunkStateImpl(const float *const input, const int32_t heads,
                               const int32_t size, float *const output) {
  const uint64_t count = static_cast<uint64_t>(heads) * size * size;
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const uint64_t head = index / (size * size);
    const int32_t row = index / size % size;
    const int32_t column = index % size;
    output[index] = input[head * size * size + column * size + row];
  }
}

template <typename T>
__global__ void SubtractChunkImpl(const int32_t padded, const int32_t heads, const int32_t size,
                                  const int32_t chunk, float *const workspace) {
  const uint64_t values = static_cast<uint64_t>(heads) * padded * size;
  const int32_t head = blockIdx.x;
  for (int32_t index = threadIdx.x; index < 64 * size; index += blockDim.x) {
    const uint64_t offset = (static_cast<uint64_t>(head) * padded + chunk * 64) * size + index;
    workspace[4 * values + offset] =
        workspace[3 * values + offset] - workspace[4 * values + offset];
  }
}

template <typename T>
__global__ void FinishChunkImpl(const int32_t tokens, const int32_t heads, const int32_t size,
                                const int32_t chunk, float *const workspace,
                                T *const output) {
  const int32_t padded = (tokens + 63) / 64 * 64;
  const uint64_t values = static_cast<uint64_t>(heads) * padded * size;
  const uint64_t matrices = static_cast<uint64_t>(heads) * padded * 64;
  const uint64_t state_offset = 7 * values + 2 * matrices + static_cast<uint64_t>(heads) * padded;
  const int32_t head = blockIdx.x;
  const int32_t chunks = padded / 64;
  const float decay = workspace[7 * values + 2 * matrices + (head * chunks + chunk) * 64];
  for (int32_t index = threadIdx.x; index < size * size; index += blockDim.x) {
    const uint64_t offset = static_cast<uint64_t>(head) * size * size + index;
    workspace[state_offset + offset] =
        __fmul_rn(workspace[state_offset + offset], decay) +
        workspace[state_offset + static_cast<uint64_t>(heads) * size * size + offset];
  }
  for (int32_t index = threadIdx.x; index < 64 * size; index += blockDim.x) {
    const int32_t token = chunk * 64 + index / size;
    if (token < tokens) {
      const uint64_t offset = (static_cast<uint64_t>(head) * padded + chunk * 64) * size + index;
      output[(static_cast<uint64_t>(token) * heads + head) * size + index % size] =
          FromFloat<T>(workspace[5 * values + offset] + workspace[6 * values + offset]);
    }
  }
}

__device__ inline void ReduceStateColumns(float *const values, float *const scratch) {
  const int32_t column = threadIdx.x * 4;
  const int32_t row = threadIdx.y;
  for (int32_t part = 0; part < 4; ++part) {
    scratch[row * 128 + column + part] = values[part];
  }
  __syncthreads();
  for (int32_t distance = blockDim.y / 2; distance > 0; distance >>= 1) {
    if (row < distance) {
      for (int32_t part = 0; part < 4; ++part) {
        values[part] += scratch[(row + distance) * 128 + column + part];
        scratch[row * 128 + column + part] = values[part];
      }
    }
    __syncthreads();
  }
  for (int32_t part = 0; part < 4; ++part) {
    values[part] = scratch[column + part];
  }
  __syncthreads();
}

template <typename T>
__global__ void ReferenceDeltaDecodeImpl(const T *const qkv, const T *const alpha,
                                         const T *const beta, const float *const decay_weight,
                                         const float *const time_bias, const int32_t key_heads,
                                         const int32_t heads, const float epsilon,
                                         const bool is_log_decay, float *const state,
                                         T *const output) {
  const int32_t head = blockIdx.x;
  const int32_t lane = threadIdx.x;
  const int32_t row = threadIdx.y;
  const int32_t key_offset = head % key_heads * 128;
  const int32_t key_size = key_heads * 128;
  __shared__ float query[128];
  __shared__ float key[128];
  __shared__ float scratch[4 * 128];
  if (row == 0) {
    float q[4];
    float k[4];
    float query_square = 0.0F;
    float key_square = 0.0F;
    for (int32_t part = 0; part < 4; ++part) {
      q[part] = ToFloat(qkv[key_offset + lane * 4 + part]);
      k[part] = ToFloat(qkv[key_size + key_offset + lane * 4 + part]);
      query_square += __fmul_rn(q[part], q[part]);
      key_square += __fmul_rn(k[part], k[part]);
    }
    query_square = __shfl_sync(0xFFFFFFFFU, WarpReduceSum(query_square), 0);
    key_square = __shfl_sync(0xFFFFFFFFU, WarpReduceSum(key_square), 0);
    const float query_inverse = rsqrtf(query_square + epsilon);
    const float key_inverse = rsqrtf(key_square + epsilon);
    for (int32_t part = 0; part < 4; ++part) {
      query[lane * 4 + part] =
          __fmul_rn(q[part], query_inverse) * static_cast<float>(1.0 / sqrt(128.0));
      key[lane * 4 + part] = k[part] * key_inverse;
    }
  }
  __syncthreads();
  const float biased = ToFloat(alpha[head]) + time_bias[head];
  const float softplus = biased > 20.0F ? biased : log1pf(expf(biased));
  const float decay_scale = is_log_decay ? -expf(decay_weight[head]) : decay_weight[head];
  const float decay = expf(decay_scale * softplus);
  const float update = ToFloat(FromFloat<T>(1.0F / (1.0F + expf(-ToFloat(beta[head])))));
  float matrix[4][32];
  float projection[4] = {};
  for (int32_t column = 0; column < 4; ++column) {
    float partials[4] = {};
    for (int32_t part = 0; part < 32; ++part) {
      const int32_t dimension = row + part * 4;
      const uint64_t index = (static_cast<uint64_t>(head) * 128 + lane * 4 + column) * 128 +
                             dimension;
      matrix[column][part] = __fmul_rn(state[index], decay);
      partials[part % 4] += __fmul_rn(matrix[column][part], key[dimension]);
    }
    projection[column] = ((partials[0] + partials[1]) + partials[2]) + partials[3];
  }
  ReduceStateColumns(projection, scratch);
  float result[4] = {};
  for (int32_t column = 0; column < 4; ++column) {
    const float value = ToFloat(qkv[2 * key_size + head * 128 + lane * 4 + column]);
    const float delta = (value - projection[column]) * update;
    float partials[4] = {};
    for (int32_t part = 0; part < 32; ++part) {
      const int32_t dimension = row + part * 4;
      const uint64_t index = (static_cast<uint64_t>(head) * 128 + lane * 4 + column) * 128 +
                             dimension;
      const float current = matrix[column][part] + __fmul_rn(key[dimension], delta);
      state[index] = current;
      partials[part % 4] += __fmul_rn(current, query[dimension]);
    }
    result[column] = ((partials[0] + partials[1]) + partials[2]) + partials[3];
  }
  ReduceStateColumns(result, scratch);
  if (row == 0) {
    for (int32_t column = 0; column < 4; ++column) {
      output[head * 128 + lane * 4 + column] = FromFloat<T>(result[column]);
    }
  }
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
    float update = __fdividef(1.0F, 1.0F + __expf(-ToFloat(beta[gate_index])));
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      update = ToFloat(FromFloat<T>(update));
    }
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
    float scale = __fdividef(1.0F, 1.0F + __expf(-ToFloat(gate[index])));
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      scale = ToFloat(FromFloat<T>(scale));
    }
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
    const float first = ToFloat(input[index]);
    const float second = ToFloat(input[index + rotary_dim / 2]);
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      const float frequency = __fdiv_rn(1.0F, powf(theta_scale, 2.0F * dimension / rotary_dim));
      const float angle =
          static_cast<float>(positions[static_cast<uint64_t>(axis) * tokens + token]) * frequency;
      const float cosine = ToFloat(FromFloat<T>(cosf(angle)));
      const float sine = ToFloat(FromFloat<T>(sinf(angle)));
      output[index] = FromFloat<T>(ToFloat(FromFloat<T>(first * cosine)) -
                                   ToFloat(FromFloat<T>(second * sine)));
      output[index + rotary_dim / 2] = FromFloat<T>(ToFloat(FromFloat<T>(first * sine)) +
                                                    ToFloat(FromFloat<T>(second * cosine)));
      return;
    }
    const float angle =
        static_cast<float>(positions[static_cast<uint64_t>(axis) * tokens + token]) *
        __powf(theta_scale, static_cast<float>(dimension));
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
