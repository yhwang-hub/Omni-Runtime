// rows == 1: columns groups of 256. Otherwise: local (16,16),
// global (round_up(columns,16), round_up(rows,16)); weight is [columns,inner].
#define OMNI_LINEAR_BASE_ARGS                                                                      \
  __global uchar *input_buffer, ulong input_offset, __global uchar *weight_buffer,                 \
      ulong weight_offset, int rows, int inner, int columns, __global uchar *output_buffer,        \
      ulong output_offset

#ifdef OMNI_PARALLEL_LINEAR
#define OMNI_LINEAR_ARGS                                                                           \
  OMNI_LINEAR_BASE_ARGS, __global uchar *second_weight_buffer, ulong second_weight_offset,         \
      int second_columns, __global uchar *second_output_buffer, ulong second_output_offset,        \
      __global uchar *third_weight_buffer, ulong third_weight_offset, int third_columns,           \
      __global uchar *third_output_buffer, ulong third_output_offset
#define OMNI_SELECT_PROJECTION                                                                     \
  do {                                                                                             \
    const int branch = get_group_id(2);                                                            \
    if (branch != 0) {                                                                             \
      weight_buffer = branch == 1 ? second_weight_buffer : third_weight_buffer;                    \
      weight_offset = branch == 1 ? second_weight_offset : third_weight_offset;                    \
      columns = branch == 1 ? second_columns : third_columns;                                      \
      output_buffer = branch == 1 ? second_output_buffer : third_output_buffer;                    \
      output_offset = branch == 1 ? second_output_offset : third_output_offset;                    \
    }                                                                                              \
  } while (0)
#else
#define OMNI_LINEAR_ARGS OMNI_LINEAR_BASE_ARGS
#define OMNI_SELECT_PROJECTION
#endif

__kernel void Linear(OMNI_LINEAR_ARGS) {
  OMNI_SELECT_PROJECTION;
  __local float left[16][17];
  __local float right[16][17];
  __local float scratch[256];
  if (rows == 1) {
    // One group per neuron; the first logical CUDA warp performs the dot.
    const int column = get_group_id(0);
    if (column >= columns) {
      return;
    }
    const int lane = get_local_id(0);
    float sum = 0.0f;
    const int values = 16 / sizeof(scalar_t);
    const int vectorized =
        inner % values == 0 && (input_offset & 15UL) == 0 && (weight_offset & 15UL) == 0;
    if (lane < 32) {
      float accumulators[8] = {0.0f};
      if (vectorized) {
        for (int d = lane * values; d < inner; d += 32 * values) {
          for (int item = 0; item < values; ++item) {
            accumulators[item] =
                fma(ReadScalar(OMNI_SCALAR(input), d + item),
                    ReadScalar(OMNI_SCALAR(weight), (ulong)column * inner + d + item),
                    accumulators[item]);
          }
        }
      } else {
        for (int d = lane; d < inner; d += 32) {
          accumulators[0] =
              fma(ReadScalar(OMNI_SCALAR(input), d),
                  ReadScalar(OMNI_SCALAR(weight), (ulong)column * inner + d), accumulators[0]);
        }
      }
      for (int item = 0; item < values; ++item) {
        sum += accumulators[item];
      }
    }
    sum = OmniBlockSum(sum, scratch);
    if (lane == 0) {
      WriteScalar(OMNI_SCALAR(output), column, sum);
    }
    return;
  }
  const int x = get_local_id(0);
  const int y = get_local_id(1);
  const int row = get_group_id(1) * 16 + y;
  const int column = get_group_id(0) * 16 + x;
  float accumulator = 0.0f;
  for (int first = 0; first < inner; first += 16) {
    left[y][x] = row < rows && first + x < inner
                     ? ReadScalar(OMNI_SCALAR(input), (ulong)row * inner + first + x)
                     : 0.0f;
    right[y][x] = column < columns && first + y < inner
                      ? ReadScalar(OMNI_SCALAR(weight), (ulong)column * inner + first + y)
                      : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < 16; ++k) {
      accumulator = fma(left[y][k], right[k][x], accumulator);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (row < rows && column < columns) {
    WriteScalar(OMNI_SCALAR(output), (ulong)row * columns + column, accumulator);
  }
}

#if defined(OMNI_NVIDIA_MMA) && (OMNI_DTYPE == 1 || OMNI_DTYPE == 2)
#ifndef OMNI_LINEAR_TILE
#define OMNI_LINEAR_TILE 0
#endif
#if OMNI_LINEAR_TILE == 1
#define OMNI_LINEAR_ROWS 128
#else
#define OMNI_LINEAR_ROWS 64
#endif
#define OMNI_LINEAR_COLUMNS 64

inline void OmniLinearStage(__global const scalar_t *source, __local ushort *target, int first_row,
                            int rows, int inner, int first, int tile_rows, ulong byte_offset) {
  const int id = get_local_id(0);
  if (inner % 8 == 0 && (byte_offset & 15UL) == 0) {
    for (int index = id * 8; index < tile_rows * 32; index += get_local_size(0) * 8) {
      const int row = index / 32;
      const int d = first + index % 32;
      const uint4 value =
          first_row + row < rows && d < inner
              ? *(__global const uint4 *)(source + (ulong)(first_row + row) * inner + d)
              : (uint4)(0);
      *(__local uint4 *)(target + row * 40 + index % 32) = value;
    }
  } else {
    for (int index = id * 2; index < tile_rows * 32; index += get_local_size(0) * 2) {
      const int row = index / 32;
      const int d = first + index % 32;
      const ushort2 value =
          first_row + row < rows
              ? OmniLoadScalarPair(source, (ulong)(first_row + row) * inner + d, inner - d)
              : (ushort2)(0);
      vstore2(value, 0, target + row * 40 + index % 32);
    }
  }
}

// Four or eight warps, each owning 16 rows and all columns of the output tile.
__attribute__((reqd_work_group_size(OMNI_LINEAR_ROWS * 2, 1, 1))) __kernel void
LinearTensor(OMNI_LINEAR_ARGS) {
  OMNI_SELECT_PROJECTION;
  __local ushort inputs[OMNI_LINEAR_ROWS * 40] __attribute__((aligned(16)));
  __local ushort weights[OMNI_LINEAR_COLUMNS * 40] __attribute__((aligned(16)));
  const int id = get_local_id(0);
  const int lane = id % 32;
  const int warp = id / 32;
  const int first_row = get_group_id(0) * OMNI_LINEAR_ROWS;
  const int first_column = get_group_id(1) * OMNI_LINEAR_COLUMNS;
  if (first_column >= columns) {
    return;
  }
  float4 accumulators[OMNI_LINEAR_COLUMNS / 8];
  for (int tile = 0; tile < OMNI_LINEAR_COLUMNS / 8; ++tile) {
    accumulators[tile] = (float4)(0.0f);
  }
  for (int first = 0; first < inner; first += 32) {
#if OMNI_LINEAR_TILE == 0
    for (int index = id * 2; index < 64 * 32; index += 256) {
      const int row = index / 32;
      const int d = first + index % 32;
      const int offset = row * 40 + index % 32;
      const ushort2 left = first_row + row < rows
                               ? OmniLoadScalarPair(OMNI_SCALAR(input),
                                                    (ulong)(first_row + row) * inner + d, inner - d)
                               : (ushort2)(0);
      const ushort2 right =
          first_column + row < columns
              ? OmniLoadScalarPair(OMNI_SCALAR(weight), (ulong)(first_column + row) * inner + d,
                                   inner - d)
              : (ushort2)(0);
      vstore2(left, 0, inputs + offset);
      vstore2(right, 0, weights + offset);
    }
#else
    OmniLinearStage(OMNI_SCALAR(input), inputs, first_row, rows, inner, first, OMNI_LINEAR_ROWS,
                    input_offset);
    OmniLinearStage(OMNI_SCALAR(weight), weights, first_column, columns, inner, first,
                    OMNI_LINEAR_COLUMNS, weight_offset);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int d = 0; d < 32; d += 16) {
      const uint4 a = OmniTensorLeft(inputs, warp, lane, d);
#pragma unroll
      for (int tile = 0; tile < OMNI_LINEAR_COLUMNS / 8; ++tile) {
        accumulators[tile] =
            OmniTensorMma(a, OmniTensorRight(weights, tile, lane, d), accumulators[tile]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#pragma unroll
  for (int tile = 0; tile < OMNI_LINEAR_COLUMNS / 8; ++tile) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int row = first_row + warp * 16 + lane / 4 + item / 2 * 8;
      const int column = first_column + tile * 8 + (lane % 4) * 2 + item % 2;
      if (row < rows && column < columns) {
        WriteScalar(OMNI_SCALAR(output), (ulong)row * columns + column, accumulators[tile][item]);
      }
    }
  }
}

#if defined(OMNI_NVIDIA_WARP)
// Eight independent warp dots per group; same argument ABI as Linear.
__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void LinearVector(OMNI_LINEAR_ARGS) {
  OMNI_SELECT_PROJECTION;
  const int lane = get_local_id(0) % 32;
  const int column = get_group_id(0) * 8 + get_local_id(0) / 32;
  if (column >= columns) {
    return;
  }
  float accumulators[8] = {0.0f};
  if (inner % 8 == 0) {
    const int aligned = (input_offset & 15UL) == 0 && (weight_offset & 15UL) == 0;
    for (int d = lane * 8; d < inner; d += 256) {
      const __global scalar_t *input = OMNI_SCALAR(input) + d;
      const __global scalar_t *weight = OMNI_SCALAR(weight) + (ulong)column * inner + d;
      const uint4 left = aligned ? *(__global const uint4 *)input : as_uint4(vload8(0, input));
      const uint4 right = aligned ? *(__global const uint4 *)weight : as_uint4(vload8(0, weight));
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        accumulators[2 * item] = fma(OmniPackedScalar(left[item]), OmniPackedScalar(right[item]),
                                     accumulators[2 * item]);
        accumulators[2 * item + 1] =
            fma(OmniPackedScalar(left[item] >> 16), OmniPackedScalar(right[item] >> 16),
                accumulators[2 * item + 1]);
      }
    }
  } else {
    for (int d = lane; d < inner; d += 32) {
      accumulators[0] =
          fma(ReadScalar(OMNI_SCALAR(input), d),
              ReadScalar(OMNI_SCALAR(weight), (ulong)column * inner + d), accumulators[0]);
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    sum += accumulators[item];
  }
  sum = OmniWarpSum(sum);
  if (lane == 0) {
    WriteScalar(OMNI_SCALAR(output), column, sum);
  }
}
#endif
#endif

#undef OMNI_SELECT_PROJECTION
#undef OMNI_LINEAR_ARGS
#undef OMNI_LINEAR_BASE_ARGS
