inline float QuantFp8Scale(uint code) {
#ifdef OMNI_NVIDIA_FP8
  if (code < 127U) {
    uint packed;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packed) : "h"((ushort)code));
    return OmniHalfToFloat((ushort)packed) * 0.5f;
  }
#endif
  const uint exponent = (code >> 3) & 15;
  const uint mantissa = code & 7;
  return code == 0 || code == 127 ? 0.0f
         : exponent == 0          ? (float)mantissa * 0.0009765625f
                                  : as_float(((exponent + 119) << 23) | (mantissa << 20));
}

inline int QuantFp4Value(uint code) {
  const int magnitude = (0xC8643210U >> ((code & 7) * 4)) & 15;
  return code & 8 ? -magnitude : magnitude;
}

inline float QuantWeight(__global const uchar *data, int type, ulong index) {
  if (type == 0) {
    return ((__global const float *)data)[index];
  }
  if (type == 1) {
    return OmniHalfToFloat(((__global const ushort *)data)[index]);
  }
  if (type == 30) {
    return as_float((uint)((__global const ushort *)data)[index] << 16);
  }
  if (type == 8) {
    __global const uchar *block = data + index / 32 * 34;
    return OmniHalfToFloat(*(__global const ushort *)block) *
           (float)((__global const char *)(block + 2))[index % 32];
  }
  __global const uchar *block = data + index / 64 * 36;
  const uint sub = (index % 64) / 16;
  const uint element = index % 16;
  const float factor = QuantFp8Scale(block[sub]);
  const uint packed = block[4 + sub * 8 + element % 8];
  const uint code = (element < 8 ? packed : packed >> 4) & 15;
  return (float)QuantFp4Value(code) * factor;
}

inline uchar QuantNearestFp4Bits(float value) {
  const float magnitude = fabs(value);
  const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  int best = 0;
  float distance = magnitude;
  for (int code = 1; code < 8; ++code) {
    const float error = fabs(magnitude - values[code]);
    if (error < distance || (error == distance && code % 2 == 0)) {
      distance = error;
      best = code;
    }
  }
  return (uchar)((uint)best | (signbit(value) ? 8U : 0U));
}

inline int QuantNearestFp4(float value) {
  return QuantFp4Value(QuantNearestFp4Bits(value));
}

inline int QuantNearestFp8(float value) {
#ifdef OMNI_NVIDIA_FP8
  const float clamped = clamp(value, 0.0f, 448.0f);
  ushort packed;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(packed) : "f"(clamped), "f"(0.0f));
  return packed & 255U;
#else
  const uint bits = as_uint(clamp(value, 0.0f, 448.0f));
  const int exponent = (int)((bits >> 23) & 255) - 127;
  const int center = clamp((exponent + 7) * 8 + (int)((bits >> 20) & 7), 0, 126);
  int best = 0;
  float distance = value;
  for (int code = max(0, center - 8); code <= min(126, center + 8); ++code) {
    const float error = fabs(value - 2.0f * QuantFp8Scale(code));
    if (error < distance || (error == distance && code % 2 == 0)) {
      distance = error;
      best = code;
    }
  }
  return best;
#endif
}

__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void
QuantizeActivation(__global uchar *input_buffer, ulong input_offset, int weight_type, int rows,
                   int columns, int is_decode, int is_ggml_rounding, int is_packed_nvfp4,
                   int is_packed_q8, __global uchar *output_buffer, ulong output_offset,
                   __global uchar *scales_buffer, ulong scales_offset) {
  const int row = get_group_id(0);
  const int id = get_local_id(0);
  const int lane = id % 32;
  const ulong offset = (ulong)row * columns;
  __local float scratch[256];
  __global float *scales = (__global float *)(scales_buffer + scales_offset);
  if (weight_type == 8 || is_decode) {
    for (int tile = get_group_id(1) * 256; tile < columns; tile += get_num_groups(1) * 256) {
      const int d = tile + id;
      const float value = d < columns ? ReadScalar(OMNI_SCALAR(input), offset + d) : 0.0f;
      float maximum = fabs(value);
#ifdef OMNI_NVIDIA_WARP
      for (int distance = 16; distance > 0; distance >>= 1) {
        float other;
        asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=f"(other) : "f"(maximum), "r"(distance));
        maximum = fmax(maximum, other);
      }
#else
      scratch[id] = maximum;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int distance = 16; distance > 0; distance >>= 1) {
        const float other = scratch[id ^ distance];
        barrier(CLK_LOCAL_MEM_FENCE);
        scratch[id] = fmax(scratch[id], other);
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      maximum = scratch[id];
      barrier(CLK_LOCAL_MEM_FENCE);
#endif
      if (d < columns) {
        const float inverse =
            maximum > 0.0f ? (is_ggml_rounding ? native_divide(127.0f, maximum) : 127.0f / maximum)
                           : 0.0f;
        const float scale = !is_ggml_rounding ? maximum / 127.0f
                            : is_decode       ? native_divide(maximum, 127.0f)
                                              : (maximum > 0.0f ? native_recip(inverse) : 0.0f);
        const float code =
            is_decode ? (maximum > 0.0f
                             ? round(is_ggml_rounding ? native_divide(value, scale) : value / scale)
                             : 0.0f)
                      : round(value * inverse);
        if (is_packed_q8) {
          ((__global char *)(output_buffer + output_offset))[offset + d] = (char)(int)code;
        } else {
          WriteScalar(OMNI_SCALAR(output), offset + d, code);
        }
        if (lane == 0) {
          scales[(ulong)row * (columns / 32) + d / 32] =
              is_decode ? OmniHalfToFloat(OmniFloatToHalf(scale)) : scale;
        }
      }
    }
    return;
  }
  float maximum = 0.0f;
  for (int d = id; d < columns; d += 256) {
    maximum = fmax(maximum, fabs(ReadScalar(OMNI_SCALAR(input), offset + d)));
  }
  scratch[id] = maximum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int distance = 128; distance > 0; distance >>= 1) {
    if (id < distance) {
      scratch[id] = fmax(scratch[id], scratch[id + distance]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float row_scale = is_ggml_rounding ? scratch[0] * (1.0f / 2688.0f) : scratch[0] / 2688.0f;
  const float inverse_row =
      row_scale > 0.0f ? (is_ggml_rounding ? native_recip(row_scale) : 1.0f / row_scale) : 0.0f;
  for (int first = id * 16; first < columns; first += 256 * 16) {
    float values[16];
    float local_maximum = 0.0f;
    for (int j = 0; j < 16; ++j) {
      values[j] = ReadScalar(OMNI_SCALAR(input), offset + first + j) * inverse_row;
      local_maximum = fmax(local_maximum, fabs(values[j]));
    }
    const int center =
        QuantNearestFp8(is_ggml_rounding ? local_maximum * (1.0f / 6.0f) : local_maximum / 6.0f);
    const int offsets[5] = {0, -1, 1, -2, 2};
    float best_scale = 0.0f;
    uchar best_scale_code = 0;
    float best_error = INFINITY;
    for (int candidate = 0; candidate < 5; ++candidate) {
      const int code = center + offsets[candidate];
      if (code < 0 || code > 126) {
        continue;
      }
      const float scale = QuantFp8Scale(code);
      const float inverse =
          scale > 0.0f ? (is_ggml_rounding ? native_divide(0.5f, scale) : 0.5f / scale) : 0.0f;
      float error = 0.0f;
      for (int j = 0; j < 16; ++j) {
        const float difference =
            fabs(values[j]) - fabs((float)QuantNearestFp4(values[j] * inverse)) * scale;
        error = fma(difference, difference, error);
      }
      if (error < best_error) {
        best_error = error;
        best_scale = scale;
        best_scale_code = (uchar)code;
      }
    }
    const float inverse =
        best_scale > 0.0f
            ? inverse_row * (is_ggml_rounding ? native_divide(0.5f, best_scale) : 0.5f / best_scale)
            : 0.0f;
    if (is_packed_nvfp4) {
      __global uchar *block =
          output_buffer + output_offset + ((ulong)row * (columns / 64) + first / 64) * 36;
      const int sub = first % 64 / 16;
      block[sub] = best_scale_code;
      for (int j = 0; j < 8; ++j) {
        const uchar low =
            QuantNearestFp4Bits(ReadScalar(OMNI_SCALAR(input), offset + first + j) * inverse);
        const uchar high =
            QuantNearestFp4Bits(ReadScalar(OMNI_SCALAR(input), offset + first + j + 8) * inverse);
        block[4 + sub * 8 + j] = low | (high << 4);
      }
    } else {
      for (int j = 0; j < 16; ++j) {
        WriteScalar(
            OMNI_SCALAR(output), offset + first + j,
            (float)QuantNearestFp4(ReadScalar(OMNI_SCALAR(input), offset + first + j) * inverse) *
                best_scale);
      }
    }
    scales[(ulong)row * (columns / 16) + first / 16] = row_scale;
  }
}

inline int QuantWeightCode(__global const uchar *weight, int type, ulong index) {
  if (type == 8) {
    return ((__global const char *)(weight + index / 32 * 34 + 2))[index % 32];
  }
  __global const uchar *block = weight + index / 64 * 36;
  const uint local_index = index % 64;
  const uint packed = block[4 + local_index / 16 * 8 + local_index % 8];
  return QuantFp4Value((packed >> (local_index % 16 < 8 ? 0 : 4)) & 15);
}

inline float QuantWeightScale(__global const uchar *weight, int type, ulong index) {
  return type == 8 ? OmniHalfToFloat(*(__global const ushort *)(weight + index / 32 * 34))
                   : QuantFp8Scale(weight[index / 64 * 36 + index % 64 / 16]);
}

inline int4 QuantWeightCodes4(__global const uchar *weight, int type, ulong index) {
  if (type == 8) {
    return convert_int4(as_char4(vload4(0, weight + index / 32 * 34 + 2 + index % 32)));
  }
  const uint local_index = index % 64;
  const uint packed =
      as_uint(vload4(0, weight + index / 64 * 36 + 4 + local_index / 16 * 8 + local_index % 8));
  const uint shift = local_index % 16 < 8 ? 0 : 4;
  return (int4)(QuantFp4Value((packed >> shift) & 15), QuantFp4Value((packed >> (8 + shift)) & 15),
                QuantFp4Value((packed >> (16 + shift)) & 15),
                QuantFp4Value((packed >> (24 + shift)) & 15));
}

__kernel void QuantizedEmbedding(__global uchar *indices_buffer, ulong indices_offset,
                                 __global uchar *weight_buffer, ulong weight_offset,
                                 int weight_type, int token_count, int hidden_size,
                                 int vocabulary_size, __global uchar *output_buffer,
                                 ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < (ulong)token_count * hidden_size) {
    const int token = OMNI_INT(indices)[index / hidden_size];
    WriteScalar(OMNI_SCALAR(output), index,
                token >= 0 && token < vocabulary_size
                    ? QuantWeight(weight_buffer + weight_offset, weight_type,
                                  (ulong)token * hidden_size + index % hidden_size)
                    : NAN);
  }
}

#define QUANT_LINEAR_ARGS                                                                          \
  __global uchar *input_buffer, ulong input_offset, __global uchar *weight_buffer,                 \
      ulong weight_offset, int weight_type, float weight_scale, int rows, int inner, int columns,  \
      __global uchar *output_buffer, ulong output_offset

__kernel void QuantizedLinear(QUANT_LINEAR_ARGS) {
  const int column = get_group_id(0);
  const int row = get_group_id(1);
  const int id = get_local_id(0);
  float sum = 0.0f;
  for (int d = id; d < inner; d += 128) {
    sum = fma(ReadScalar(OMNI_SCALAR(input), (ulong)row * inner + d),
              QuantWeight(weight_buffer + weight_offset, weight_type, (ulong)column * inner + d),
              sum);
  }
  __local float scratch[128];
  sum = OmniBlockSum(sum, scratch);
  if (id == 0) {
    WriteScalar(OMNI_SCALAR(output), (ulong)row * columns + column, sum * weight_scale);
  }
}

__kernel void QuantizedScaledLinear(QUANT_LINEAR_ARGS, __global uchar *activation_scales_buffer,
                                    ulong activation_scales_offset, int activation_block_size) {
  const int column = get_group_id(0);
  const int row = get_group_id(1);
  const int id = get_local_id(0);
  const int block_size = weight_type == 8 ? 32 : 16;
  __global const uchar *weight = weight_buffer + weight_offset;
  __global const float *scales =
      (__global const float *)(activation_scales_buffer + activation_scales_offset);
  float sum = 0.0f;
  for (int first = id * 4; first < inner; first += 128 * 4) {
    if (activation_block_size == 16) {
      for (int j = 0; j < 4; ++j) {
#ifdef OMNI_PACKED_NVFP4
        const float value =
            QuantWeight(input_buffer + input_offset, 40, (ulong)row * inner + first + j);
#else
        const float value = ReadScalar(OMNI_SCALAR(input), (ulong)row * inner + first + j);
#endif
        sum = fma(value, QuantWeight(weight, weight_type, (ulong)column * inner + first + j), sum);
      }
      continue;
    }
    int dot = 0;
    const int4 weights = QuantWeightCodes4(weight, weight_type, (ulong)column * inner + first);
    for (int j = 0; j < 4; ++j) {
      dot += (int)ReadScalar(OMNI_SCALAR(input), (ulong)row * inner + first + j) * weights[j];
    }
    const float scale =
        QuantWeightScale(weight, weight_type, (ulong)column * inner + first) *
        scales[(ulong)row * (inner / activation_block_size) + first / activation_block_size];
    sum = fma((float)dot, scale, sum);
  }
  __local float scratch[128];
  sum = OmniBlockSum(sum, scratch);
  if (id == 0) {
    if (activation_block_size == 16) {
      sum *= scales[(ulong)row * (inner / 16)];
    }
    WriteScalar(OMNI_SCALAR(output), (ulong)row * columns + column, sum * weight_scale);
  }
}

#if defined(OMNI_NVIDIA_MMA)
inline uint QuantPackedWord(__global const uchar *data) {
  return ((ulong)data & 3UL) == 0UL ? *(__global const uint *)data : as_uint(vload4(0, data));
}

inline uint QuantActivationCodes(__global const scalar_t *input, ulong first) {
#ifdef OMNI_PACKED_Q8
  return QuantPackedWord((__global const uchar *)input + first);
#else
  uint packed = 0;
  for (int j = 0; j < 4; ++j) {
    packed |= (uint)(uchar)(int)ReadScalar(input, first + j) << (j * 8);
  }
  return packed;
#endif
}

inline uint QuantBytePerm(uint low, uint high, uint selector) {
  uint value;
  selector &= 0x7777;
  asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(value) : "r"(low), "r"(high), "r"(selector));
  return value;
}

inline uint2 QuantFp4Codes2(uint packed) {
  uint values[2];
  const uint selector = 0x32103210U | ((packed & 0x88888888U) >> 1);
  for (uint part = 0; part < 2; ++part) {
    const uint codes = packed >> (part * 16);
    const uint low = QuantBytePerm(0x03020100U, 0x0C080604U, codes);
    const uint high = QuantBytePerm(0xFDFEFF00U, 0xF4F8FAFCU, codes);
    values[part] = QuantBytePerm(low, high, selector >> (part * 16));
  }
  return (uint2)(QuantBytePerm(values[0], values[1], 0x6420),
                 QuantBytePerm(values[0], values[1], 0x7531));
}

inline int QuantDot4(uint x, uint w, int sum) {
  asm volatile("dp4a.s32.s32 %0, %1, %2, %0;" : "+r"(sum) : "r"(x), "r"(w));
  return sum;
}

inline int QuantSharedActivationWord(int word, int mask) {
  return word ^ ((word >> 5) & mask);
}

__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void
QuantizedScaledGemv(QUANT_LINEAR_ARGS, __global uchar *activation_scales_buffer,
                    ulong activation_scales_offset, int activation_block_size,
                    __local uint *packed_input) {
  const int id = get_local_id(0);
#ifdef OMNI_GEMV_WARPS
  const int warps = OMNI_GEMV_WARPS;
#else
  const int warps = rows <= 4 ? 4 : 2;
#endif
  const int column = get_group_id(0) * (8 / warps) + id / (32 * warps);
  const int row = get_group_id(1);
  const int lane = id % 32;
  const int warp = id / 32 % warps;
#ifdef OMNI_GEMV_TYPE
  const int type = OMNI_GEMV_TYPE;
#else
  const int type = weight_type;
#endif
  const int values = type == 8 ? 8 : 32;
  const int swizzle_mask = inner % 128 == 0 ? 31 : 0;
#ifdef OMNI_PACKED_Q8
  const int has_shared_input = 0;
#else
  const int has_shared_input = inner <= 32768;
#endif
  __global const uchar *weight = weight_buffer + weight_offset;
  __global const float *scales =
      (__global const float *)(activation_scales_buffer + activation_scales_offset);
  if (has_shared_input) {
    for (int d = id * 4; d < inner; d += 256 * 4) {
      packed_input[QuantSharedActivationWord(d / 4, swizzle_mask)] =
          QuantActivationCodes(OMNI_SCALAR(input), (ulong)row * inner + d);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float sum = 0.0f;
  for (int first = (warp * 32 + lane) * values; column < columns && first < inner;
       first += 32 * warps * values) {
    if (type == 8) {
      const int future = first + 2 * 32 * warps * values;
      if (future < inner) {
        const ulong address = (ulong)(weight + ((ulong)column * inner + future) / 32 * 34);
        asm volatile("prefetch.global.L2 [%0];" : : "l"(address));
      }
    }
    float partial = 0.0f;
    for (int sub = 0; sub < values; sub += type == 8 ? 8 : 16) {
      int dot = 0;
      if (type == 8) {
        for (int group = 0; group < 8; group += 4) {
          const int d = first + group;
          const uint w = as_uint(
              vload2(0, (__global const ushort *)(weight + (ulong)column * (inner / 32 * 34) +
                                                  d / 32 * 34 + 2 + d % 32)));
          const uint x = has_shared_input
                             ? packed_input[QuantSharedActivationWord(d / 4, swizzle_mask)]
                             : QuantActivationCodes(OMNI_SCALAR(input), (ulong)row * inner + d);
          dot = QuantDot4(x, w, dot);
        }
      } else {
        const ulong index = (ulong)column * inner + first + sub;
        __global const uchar *codes = weight + index / 64 * 36 + 4 + index % 64 / 16 * 8;
        for (int group = 0; group < 2; ++group) {
          const uint2 w = QuantFp4Codes2(QuantPackedWord(codes + group * 4));
          const int d = first + sub + group * 4;
          const uint low = has_shared_input
                               ? packed_input[QuantSharedActivationWord(d / 4, swizzle_mask)]
                               : QuantActivationCodes(OMNI_SCALAR(input), (ulong)row * inner + d);
          const uint high =
              has_shared_input
                  ? packed_input[QuantSharedActivationWord(d / 4 + 2, swizzle_mask)]
                  : QuantActivationCodes(OMNI_SCALAR(input), (ulong)row * inner + d + 8);
          dot = QuantDot4(low, w.s0, dot);
          dot = QuantDot4(high, w.s1, dot);
        }
      }
      const float scale = QuantWeightScale(weight, type, (ulong)column * inner + first + sub) *
                          scales[(ulong)row * (inner / 32) + (first + sub) / 32];
      if (type == 8) {
        sum = fma((float)dot, scale, sum);
      } else {
        partial = fma((float)dot, scale, partial);
      }
    }
    if (type != 8) {
      sum += partial;
    }
  }
  __local float partials[256];
  partials[id] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (warp != 0) {
    return;
  }
  for (int part = 1; part < warps; ++part) {
    sum += partials[id + part * 32];
  }
  sum = OmniWarpSum(sum);
  if (lane == 0 && column < columns) {
    WriteScalar(OMNI_SCALAR(output), (ulong)row * columns + column, sum * weight_scale);
  }
}

inline float4 QuantMma(uint4 a, uint2 b, float4 accumulator, int is_bfloat) {
  float c0 = accumulator.s0;
  float c1 = accumulator.s1;
  float c2 = accumulator.s2;
  float c3 = accumulator.s3;
  if (is_bfloat) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a.s0), "r"(a.s1), "r"(a.s2), "r"(a.s3), "r"(b.s0), "r"(b.s1));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a.s0), "r"(a.s1), "r"(a.s2), "r"(a.s3), "r"(b.s0), "r"(b.s1));
  }
  return (float4)(c0, c1, c2, c3);
}

inline uint QuantPair(__local const ushort *values, int index) {
  return as_uint(vload2(0, values + index));
}

inline uint4 QuantLoadLeft(__local const ushort *values, int stride, int row, int lane, int d) {
  const ulong address = (ulong)(values + (row + lane % 16) * stride + d + lane / 16 * 8);
  uint4 result;
  asm volatile("{ .reg .u64 shared_address; cvta.to.shared.u64 shared_address, %4; "
               "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [shared_address]; }"
               : "=r"(result.s0), "=r"(result.s1), "=r"(result.s2), "=r"(result.s3)
               : "l"(address));
  return result;
}

inline uint2 QuantLoadRight(__local const ushort *values, int stride, int tile, int lane, int d) {
  const ulong address = (ulong)(values + (tile * 8 + lane % 8) * stride + d + lane / 8 % 2 * 8);
  uint2 result;
  asm volatile("{ .reg .u64 shared_address; cvta.to.shared.u64 shared_address, %2; "
               "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [shared_address]; }"
               : "=r"(result.s0), "=r"(result.s1)
               : "l"(address));
  return result;
}

inline int4 QuantMmaInt(uint4 a, uint2 b) {
  int c0;
  int c1;
  int c2;
  int c3;
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {0,0,0,0};"
               : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
               : "r"(a.s0), "r"(a.s1), "r"(a.s2), "r"(a.s3), "r"(b.s0), "r"(b.s1));
  return (int4)(c0, c1, c2, c3);
}

#if defined(OMNI_GGML_SPLIT_K) && !defined(OMNI_GGML_LARGE_SPLIT_TILE)
#define QUANT_Q8_ROWS 8
#define QUANT_Q8_SHARED_ROWS 16
#define QUANT_Q8_TILES 2
#else
#define QUANT_Q8_ROWS 64
#define QUANT_Q8_SHARED_ROWS 64
#define QUANT_Q8_TILES 8
#endif
__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void
QuantizedQ8Tensor(QUANT_LINEAR_ARGS, __global uchar *activation_scales_buffer,
                  ulong activation_scales_offset, int activation_block_size, int token_tile,
                  int stream_groups) {
  __local uint left[QUANT_Q8_SHARED_ROWS * 12] __attribute__((aligned(16)));
  __local uint right[64 * 12] __attribute__((aligned(16)));
  __local float input_scales[QUANT_Q8_SHARED_ROWS];
  __local float weight_scales[64];
  __global const float *scales =
      (__global const float *)(activation_scales_buffer + activation_scales_offset);
  __global const uchar *weight = weight_buffer + weight_offset;
  const int id = get_local_id(0);
  const int lane = id % 32;
  const int warp = id / 32;
  const int warp_row = QUANT_Q8_ROWS == 8 ? 0 : warp * 16;
  const int warp_column = QUANT_Q8_ROWS == 8 ? warp * 16 : 0;
  const int row_start = get_group_id(0) * QUANT_Q8_ROWS;
  const int column_start = get_group_id(1) * 64;
  float4 accumulators[QUANT_Q8_TILES];
#ifdef OMNI_GGML_SPLIT_K
  float4 last[QUANT_Q8_TILES];
  float4 previous[QUANT_Q8_TILES];
  for (int tile = 0; tile < QUANT_Q8_TILES; ++tile) {
    last[tile] = (float4)(0.0f);
    previous[tile] = (float4)(0.0f);
  }
  int has_last = 0;
  const long blocks = inner / 32;
  const long token_tiles = (rows - 1) / token_tile + 1;
  const long output_tile = column_start / 128 * token_tiles + row_start / token_tile;
  const long total_blocks = ((columns - 1) / 128 + 1) * token_tiles * blocks;
  const int first_part = (int)(output_tile * blocks * stream_groups / total_blocks);
  const int last_part = (int)min((long)(stream_groups - 1),
                                 (output_tile + 1) * blocks * stream_groups / total_blocks);
#else
  const int first_part = 0;
  const int last_part = 0;
#endif
  for (int part = last_part; part >= first_part; --part) {
    int first_inner = 0;
    int stop_inner = inner;
#ifdef OMNI_GGML_SPLIT_K
    long first = (long)part * total_blocks / stream_groups;
    long stop = (long)(part + 1) * total_blocks / stream_groups;
    first -= first % blocks % 8;
    stop -= stop % blocks % 8;
    first_inner = (int)(max(first, output_tile * blocks) - output_tile * blocks) * 32;
    stop_inner = (int)(min(stop, (output_tile + 1) * blocks) - output_tile * blocks) * 32;
#endif
    if (first_inner >= stop_inner) {
      continue;
    }
    for (int tile = 0; tile < QUANT_Q8_TILES; ++tile) {
      accumulators[tile] = (float4)(0.0f);
    }
    for (int first = first_inner; first < stop_inner; first += 32) {
      for (int index = id; index < 64 * 8; index += 128) {
        const int r = index / 8;
        const int local_index = index % 8;
        const int d = first + local_index * 4;
        if (r < QUANT_Q8_SHARED_ROWS) {
          left[r * 12 + local_index] =
              row_start + r < rows && d < stop_inner
                  ? QuantActivationCodes(OMNI_SCALAR(input), (ulong)(row_start + r) * inner + d)
                  : 0;
        }
        right[r * 12 + local_index] =
            column_start + r < columns && d < stop_inner
                ? as_uint(vload2(
                      0, (__global const ushort
                              *)(weight + ((ulong)(column_start + r) * (inner / 32) + d / 32) * 34 +
                                 2 + d % 32)))
                : 0;
      }
      if (id < QUANT_Q8_SHARED_ROWS) {
        input_scales[id] = row_start + id < rows
                               ? scales[(ulong)(row_start + id) * (inner / 32) + first / 32]
                               : 0.0f;
      }
      if (id < 64) {
        weight_scales[id] =
            column_start + id < columns
                ? QuantWeightScale(weight, 8, (ulong)(column_start + id) * inner + first)
                : 0.0f;
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      const uint4 a = QuantLoadLeft((__local const ushort *)left, 24, warp_row, lane, 0);
      for (int tile = 0; tile < QUANT_Q8_TILES; ++tile) {
        const uint2 b =
            QuantLoadRight((__local const ushort *)right, 24, warp_column / 8 + tile, lane, 0);
        const int4 partial = QuantMmaInt(a, b);
        for (int item = 0; item < 4; ++item) {
          const int row = warp_row + lane / 4 + item / 2 * 8;
          const int column = warp_column + tile * 8 + lane % 4 * 2 + item % 2;
          const float weighted = (float)partial[item] * weight_scales[column];
          accumulators[tile][item] = fma(weighted, input_scales[row], accumulators[tile][item]);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
#ifdef OMNI_GGML_SPLIT_K
    for (int tile = 0; tile < QUANT_Q8_TILES; ++tile) {
      if (!has_last) {
        last[tile] = accumulators[tile];
      } else {
        previous[tile] += accumulators[tile];
      }
    }
    has_last = 1;
#endif
  }
  for (int tile = 0; tile < QUANT_Q8_TILES; ++tile) {
    for (int item = 0; item < 4; ++item) {
      const int local_row = warp_row + lane / 4 + item / 2 * 8;
      const int row = row_start + local_row;
      const int column = column_start + warp_column + tile * 8 + lane % 4 * 2 + item % 2;
      if (local_row < QUANT_Q8_ROWS && row < rows && column < columns) {
#ifdef OMNI_GGML_SPLIT_K
        const float sum = last[tile][item] + previous[tile][item];
#else
        const float sum = accumulators[tile][item];
#endif
        WriteScalar(OMNI_SCALAR(output), (ulong)row * columns + column, sum * weight_scale);
      }
    }
  }
}

#ifdef OMNI_PACKED_NVFP4
inline void QuantLoadFp4Tile(__global const uchar *source, int first_row, int valid_rows,
                             int tile_rows, int inner, int first, __local ushort *tile,
                             __local int2 *exponent_ranges) {
  for (int index = get_local_id(0); index < tile_rows * 4; index += get_local_size(0)) {
    const int row = index / 4;
    const int sub = index % 4;
    __local ushort *destination = tile + row * 72 + sub * 16;
    if (row >= valid_rows) {
      vstore16((ushort16)0, 0, destination);
      if (sub == 0) {
        exponent_ranges[row] = (int2)(31, 0);
      }
      continue;
    }
    __global const uchar *block = source + ((ulong)(first_row + row) * inner + first) / 64 * 36;
    if (sub == 0) {
      const uint packed_scales = QuantPackedWord(block);
      int minimum = 31;
      int maximum = 0;
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        const uint code = (packed_scales >> (group * 8)) & 255U;
        if (code != 0U && code != 127U) {
          const int exponent = max(1, (int)((code >> 3) & 15U));
          minimum = min(minimum, exponent);
          maximum = max(maximum, exponent);
        }
      }
      exponent_ranges[row] = (int2)(minimum, maximum);
    }
    const float scale = QuantFp8Scale(block[sub]);
    const uint2 first_codes = QuantFp4Codes2(QuantPackedWord(block + 4 + sub * 8));
    const uint2 second_codes = QuantFp4Codes2(QuantPackedWord(block + 8 + sub * 8));
    const uint4 codes = (uint4)(first_codes.x, second_codes.x, first_codes.y, second_codes.y);
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      const float4 values = convert_float4(as_char4(codes[group])) * scale;
      vstore4(convert_ushort4(as_uint4(values) >> 16), group, destination);
    }
  }
}

#ifdef OMNI_SMALL_FP4_TILE
#define QUANT_BFLOAT_ROWS 16
#define QUANT_BFLOAT_THREADS 32
#else
#define QUANT_BFLOAT_ROWS 128
#define QUANT_BFLOAT_THREADS 256
#endif
#define QUANT_BFLOAT_COLUMNS 32
#else
#define QUANT_BFLOAT_ROWS 64
#define QUANT_BFLOAT_THREADS 128
#define QUANT_BFLOAT_COLUMNS 128
#endif
#define QUANT_BFLOAT_TILES (QUANT_BFLOAT_COLUMNS / 8)
__attribute__((reqd_work_group_size(QUANT_BFLOAT_THREADS, 1, 1))) __kernel void
QuantizedBfloatTensor(QUANT_LINEAR_ARGS, __global uchar *activation_scales_buffer,
                      ulong activation_scales_offset, int activation_block_size
#ifdef OMNI_PACKED_NVFP4
                      ,
                      int token_tile, int stream_groups
#endif
) {
  __local ushort left[QUANT_BFLOAT_ROWS * 72] __attribute__((aligned(16)));
  __local ushort right[QUANT_BFLOAT_COLUMNS * 72] __attribute__((aligned(16)));
#ifdef OMNI_PACKED_NVFP4
  __local int2 left_exponent_ranges[QUANT_BFLOAT_ROWS];
  __local int2 right_exponent_ranges[QUANT_BFLOAT_COLUMNS];
#endif
  __global const float *scales =
      (__global const float *)(activation_scales_buffer + activation_scales_offset);
  const int id = get_local_id(0);
  const int lane = id % 32;
  const int warp = id / 32;
#if defined(OMNI_PACKED_NVFP4) && defined(OMNI_GGML_SPLIT_K)
  const int tile_rows = token_tile;
#else
  const int tile_rows = QUANT_BFLOAT_ROWS;
#endif
  const int row_start = get_group_id(0) * tile_rows;
  const int column_start = get_group_id(1) * QUANT_BFLOAT_COLUMNS;
  float4 accumulators[QUANT_BFLOAT_TILES];
#pragma unroll
  for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
    accumulators[tile] = (float4)(0.0f);
  }
#if defined(OMNI_PACKED_NVFP4) && defined(OMNI_GGML_SPLIT_K)
  float4 last[QUANT_BFLOAT_TILES];
  float4 previous[QUANT_BFLOAT_TILES];
#pragma unroll
  for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
    last[tile] = (float4)(0.0f);
    previous[tile] = (float4)(0.0f);
  }
  int has_last = 0;
  const long blocks = inner / 64;
  const long token_tiles = (rows - 1) / token_tile + 1;
  const long output_tile = (long)(column_start / 128) * token_tiles + get_group_id(0);
  const long total_blocks = ((columns - 1) / 128 + 1) * token_tiles * blocks;
  const int first_part = (int)(output_tile * blocks * stream_groups / total_blocks);
  const int last_part = (int)min((long)(stream_groups - 1),
                                 (output_tile + 1) * blocks * stream_groups / total_blocks);
  for (int part = last_part; part >= first_part; --part) {
    long first_block = (long)part * total_blocks / stream_groups;
    long stop_block = (long)(part + 1) * total_blocks / stream_groups;
    first_block -= first_block % blocks % 8;
    stop_block -= stop_block % blocks % 8;
    const int first_inner =
        (int)(max(first_block, output_tile * blocks) - output_tile * blocks) * 64;
    const int stop_inner =
        (int)(min(stop_block, (output_tile + 1) * blocks) - output_tile * blocks) * 64;
    if (first_inner >= stop_inner) {
      continue;
    }
#pragma unroll
    for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
      accumulators[tile] = (float4)(0.0f);
    }
#else
  const int first_inner = 0;
  const int stop_inner = inner;
#endif
    for (int first = first_inner; first < stop_inner; first += 64) {
#ifdef OMNI_PACKED_NVFP4
      QuantLoadFp4Tile(input_buffer + input_offset, row_start, min(rows - row_start, tile_rows),
                       QUANT_BFLOAT_ROWS, inner, first, left, left_exponent_ranges);
      QuantLoadFp4Tile(weight_buffer + weight_offset, column_start, columns - column_start,
                       QUANT_BFLOAT_COLUMNS, inner, first, right, right_exponent_ranges);
#else
#pragma unroll 1
    for (int index = id; index < 128 * 64; index += QUANT_BFLOAT_THREADS) {
      const int r = index / 64;
      const int local_index = index % 64;
      const int d = first + local_index;
      if (r < QUANT_BFLOAT_ROWS) {
        const float x = row_start + r < rows && d < inner
                            ? ReadScalar(OMNI_SCALAR(input), (ulong)(row_start + r) * inner + d)
                            : 0.0f;
        left[r * 72 + local_index] = OmniFloatToBfloat(x);
      }
      const float w = column_start + r < columns && d < inner
                          ? QuantWeight(weight_buffer + weight_offset, 40,
                                        (ulong)(column_start + r) * inner + d)
                          : 0.0f;
      right[r * 72 + local_index] = (ushort)(as_uint(w) >> 16);
    }
#endif
      barrier(CLK_LOCAL_MEM_FENCE);
#ifdef OMNI_PACKED_NVFP4
      const int2 first_range = left_exponent_ranges[warp * 16 + lane / 4];
      const int2 second_range = left_exponent_ranges[warp * 16 + lane / 4 + 8];
      int minimum = min(first_range.s0, second_range.s0);
      int maximum = max(first_range.s1, second_range.s1);
      int right_minimum = 31;
      int right_maximum = 0;
#pragma unroll
      for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
        const int2 first_right = right_exponent_ranges[tile * 8 + lane % 4 * 2];
        const int2 second_right = right_exponent_ranges[tile * 8 + lane % 4 * 2 + 1];
        right_minimum = min(right_minimum, min(first_right.s0, second_right.s0));
        right_maximum = max(right_maximum, max(first_right.s1, second_right.s1));
      }
      minimum += right_minimum;
      maximum += right_maximum;
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2) {
        int other;
        asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(minimum), "r"(delta));
        minimum = min(minimum, other);
        asm("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(maximum), "r"(delta));
        maximum = max(maximum, other);
      }
      // 64 * (12 * 15)^2 * 2^10 fits int32; align the two E4M3 scales at their minimum.
      if (maximum - minimum <= 10) {
        const int shift = 22 - minimum;
        const float unit = as_float((uint)(127 + shift) << 23);
        const float inverse_unit = as_float((uint)(127 - shift) << 23);
        int4 partial[QUANT_BFLOAT_TILES];
#pragma unroll
        for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
          partial[tile] = (int4)(0);
        }
#pragma unroll 1
        for (int d = 0; d < 64; d += 16) {
          const uint4 a = QuantLoadLeft(left, 72, warp * 16, lane, d);
#pragma unroll
          for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
            const uint2 b = QuantLoadRight(right, 72, tile, lane, d);
            partial[tile] += convert_int4_rtz(QuantMma(a, b, (float4)(0.0f), 1) * unit);
          }
        }
#pragma unroll
        for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
          accumulators[tile] += convert_float4_rtz(partial[tile]) * inverse_unit;
        }
      } else {
        // FP4 products are multiples of 2^-20; the exact K64 sum fits in 49 bits.
        long4 partial[QUANT_BFLOAT_TILES];
#pragma unroll
        for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
          partial[tile] = (long4)(0);
        }
#endif
#pragma unroll 1
        for (int d = 0; d < 64; d += 16) {
          const uint4 a = QuantLoadLeft(left, 72, warp * 16, lane, d);
#pragma unroll
          for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
            const uint2 b = QuantLoadRight(right, 72, tile, lane, d);
#ifdef OMNI_PACKED_NVFP4
            partial[tile] += convert_long4_rtz(QuantMma(a, b, (float4)(0.0f), 1) * 0x1.0p20f);
#else
        accumulators[tile] = QuantMma(a, b, accumulators[tile], 1);
#endif
          }
        }
#ifdef OMNI_PACKED_NVFP4
#pragma unroll
        for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
          accumulators[tile] += convert_float4_rtz(partial[tile]) * 0x1.0p-20f;
        }
      }
#endif
      barrier(CLK_LOCAL_MEM_FENCE);
    }
#if defined(OMNI_PACKED_NVFP4) && defined(OMNI_GGML_SPLIT_K)
#pragma unroll
    for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int local_row = warp * 16 + lane / 4 + item / 2 * 8;
        const int row = row_start + local_row;
        const float partial = local_row < tile_rows && row < rows
                                  ? accumulators[tile][item] * scales[(ulong)row * (inner / 16)]
                                  : 0.0f;
        if (!has_last) {
          last[tile][item] = partial;
        } else {
          previous[tile][item] += partial;
        }
      }
    }
    has_last = 1;
  }
#endif
#pragma unroll
  for (int tile = 0; tile < QUANT_BFLOAT_TILES; ++tile) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int row = row_start + warp * 16 + lane / 4 + item / 2 * 8;
      const int column = column_start + tile * 8 + lane % 4 * 2 + item % 2;
      if (row < row_start + tile_rows && row < rows && column < columns) {
#if defined(OMNI_PACKED_NVFP4) && defined(OMNI_GGML_SPLIT_K)
        const float sum = last[tile][item] + previous[tile][item];
#else
        const float sum = accumulators[tile][item] * scales[(ulong)row * (inner / 16)];
#endif
        WriteScalar(OMNI_SCALAR(output), (ulong)row * columns + column, sum * weight_scale);
      }
    }
  }
}

__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void QuantizedTensor(QUANT_LINEAR_ARGS) {
  __local ushort left[64 * 40] __attribute__((aligned(16)));
  __local ushort right[64 * 40] __attribute__((aligned(16)));
  const int id = get_local_id(0);
  const int lane = id % 32;
  const int warp = id / 32;
  const int row_start = get_group_id(1) * 64;
  const int column_start = get_group_id(0) * 64;
#ifdef OMNI_DENSE_TYPE
  const int type = OMNI_DENSE_TYPE;
#else
  const int type = weight_type;
#endif
  const int is_bfloat = type == 30 || type == 40;
  float4 accumulators[8];
  for (int tile = 0; tile < 8; ++tile) {
    accumulators[tile] = (float4)(0.0f);
  }
  for (int first = 0; first < inner; first += 32) {
    for (int index = id; index < 64 * 32; index += 128) {
      const int r = index / 32;
      const int d = first + index % 32;
      const float x = row_start + r < rows && d < inner
                          ? ReadScalar(OMNI_SCALAR(input), (ulong)(row_start + r) * inner + d)
                          : 0.0f;
      left[r * 40 + index % 32] = is_bfloat ? OmniFloatToBfloat(x) : OmniFloatToHalf(x);
      if (type == 30) {
        const ushort bits =
            column_start + r < columns && d < inner
                ? ((__global const ushort *)(weight_buffer +
                                             weight_offset))[(ulong)(column_start + r) * inner + d]
                : (ushort)0;
        right[r * 40 + index % 32] = (bits & 0x7fffU) > 0x7f80U ? (ushort)0x7fffU : bits;
      } else {
        const float w = column_start + r < columns && d < inner
                            ? QuantWeight(weight_buffer + weight_offset, type,
                                          (ulong)(column_start + r) * inner + d)
                            : 0.0f;
        right[r * 40 + index % 32] = is_bfloat ? OmniFloatToBfloat(w) : OmniFloatToHalf(w);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int d = 0; d < 32; d += 16) {
      const uint left_address =
          (uint)(ulong)(left + (warp * 16 + lane % 16) * 40 + d + lane / 16 * 8);
      uint4 a;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(a.s0), "=r"(a.s1), "=r"(a.s2), "=r"(a.s3)
                   : "r"(left_address));
      for (int tile = 0; tile < 8; ++tile) {
        const uint right_address =
            (uint)(ulong)(right + (tile * 8 + lane % 8) * 40 + d + (lane / 8 % 2) * 8);
        uint2 b;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(b.s0), "=r"(b.s1)
                     : "r"(right_address));
        accumulators[tile] = QuantMma(a, b, accumulators[tile], is_bfloat);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int tile = 0; tile < 8; ++tile) {
    for (int item = 0; item < 4; ++item) {
      const int row = row_start + warp * 16 + lane / 4 + item / 2 * 8;
      const int column = column_start + tile * 8 + (lane % 4) * 2 + item % 2;
      if (row < rows && column < columns) {
        WriteScalar(OMNI_SCALAR(output), (ulong)row * columns + column,
                    accumulators[tile][item] * weight_scale);
      }
    }
  }
}

__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void
QuantizedScaledTensor(QUANT_LINEAR_ARGS, __global uchar *activation_scales_buffer,
                      ulong activation_scales_offset, int activation_block_size) {
  __local ushort left[64 * 40];
  __local ushort right[64 * 40];
  __global const uchar *weight = weight_buffer + weight_offset;
  __global const float *scales =
      (__global const float *)(activation_scales_buffer + activation_scales_offset);
  const int id = get_local_id(0);
  const int lane = id % 32;
  const int warp = id / 32;
  const int row_start = get_group_id(1) * 64;
  const int column_start = get_group_id(0) * 64;
  const int is_native_fp4 = weight_type == 40 && activation_block_size == 16;
  const int block_size = weight_type == 8 || is_native_fp4 ? 32 : 16;
  float4 accumulators[8];
  for (int tile = 0; tile < 8; ++tile) {
    accumulators[tile] = (float4)(0.0f);
  }
  for (int first = 0; first < inner; first += block_size) {
    for (int index = id; index < 64 * 32; index += 128) {
      const int r = index / 32;
      const int local_index = index % 32;
      const int d = first + local_index;
      const float x = local_index < block_size && row_start + r < rows
                          ? ReadScalar(OMNI_SCALAR(input), (ulong)(row_start + r) * inner + d)
                          : 0.0f;
      const float w =
          local_index < block_size && column_start + r < columns
              ? is_native_fp4
                    ? QuantWeight(weight, weight_type, (ulong)(column_start + r) * inner + d)
                    : (float)QuantWeightCode(weight, weight_type,
                                             (ulong)(column_start + r) * inner + d)
              : 0.0f;
      left[r * 40 + local_index] = is_native_fp4 ? OmniFloatToBfloat(x) : OmniFloatToHalf(x);
      right[r * 40 + local_index] = is_native_fp4 ? OmniFloatToBfloat(w) : OmniFloatToHalf(w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 partial[8];
    for (int tile = 0; tile < 8; ++tile) {
      partial[tile] = is_native_fp4 ? accumulators[tile] : (float4)(0.0f);
    }
    for (int d = 0; d < block_size; d += 16) {
      const int r = warp * 16 + lane / 4;
      const int k = d + lane % 4 * 2;
      const uint4 a =
          (uint4)(QuantPair(left, r * 40 + k), QuantPair(left, (r + 8) * 40 + k),
                  QuantPair(left, r * 40 + k + 8), QuantPair(left, (r + 8) * 40 + k + 8));
      for (int tile = 0; tile < 8; ++tile) {
        const int c = tile * 8 + lane / 4;
        const uint2 b = (uint2)(QuantPair(right, c * 40 + k), QuantPair(right, c * 40 + k + 8));
        partial[tile] = QuantMma(a, b, partial[tile], is_native_fp4);
      }
    }
    for (int tile = 0; tile < 8; ++tile) {
      if (is_native_fp4) {
        accumulators[tile] = partial[tile];
        continue;
      }
      for (int item = 0; item < 4; ++item) {
        const int row = row_start + warp * 16 + lane / 4 + item / 2 * 8;
        const int column = column_start + tile * 8 + lane % 4 * 2 + item % 2;
        if (row < rows && column < columns) {
          const float scale =
              QuantWeightScale(weight, weight_type, (ulong)column * inner + first) *
              scales[(ulong)row * (inner / activation_block_size) + first / activation_block_size];
          accumulators[tile][item] = fma(partial[tile][item], scale, accumulators[tile][item]);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int tile = 0; tile < 8; ++tile) {
    for (int item = 0; item < 4; ++item) {
      const int row = row_start + warp * 16 + lane / 4 + item / 2 * 8;
      const int column = column_start + tile * 8 + lane % 4 * 2 + item % 2;
      if (row < rows && column < columns) {
        WriteScalar(OMNI_SCALAR(output), (ulong)row * columns + column,
                    (is_native_fp4 ? accumulators[tile][item] * scales[(ulong)row * (inner / 16)]
                                   : accumulators[tile][item]) *
                        weight_scale);
      }
    }
  }
}
#endif
