inline int OmniMergedPatch(int index, int grid_width, int merge_size) {
  const int area = merge_size * merge_size;
  const int group = index / area;
  const int local_index = index % area;
  const int groups_per_row = grid_width / merge_size;
  return (group / groups_per_row * merge_size + local_index / merge_size) * grid_width +
         group % groups_per_row * merge_size + local_index % merge_size;
}

#if defined(OMNI_NVIDIA_MMA) && (OMNI_DTYPE == 1 || OMNI_DTYPE == 2)
// Match VisionPatchEmbedPairTensorImpl: accumulate both frames without rounding their sum.
__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void PairedPatchEmbeddingTensor(
    __global uchar *input_buffer, ulong input_offset, __global uchar *first_weight_buffer,
    ulong first_weight_offset, __global uchar *second_weight_buffer, ulong second_weight_offset,
    __global uchar *bias_buffer, ulong bias_offset, int height, int width, int channels,
    int output_channels, int spatial_patch, __global uchar *output_buffer, ulong output_offset) {
  __local ushort inputs[64 * 40];
  __local ushort first_weights[64 * 40];
  __local ushort second_weights[64 * 40];
  const int id = get_local_id(0);
  const int lane = id % 32;
  const int warp = id / 32;
  const int patches = (height / spatial_patch) * (width / spatial_patch);
  const int inner = channels * spatial_patch * spatial_patch;
  const int first_row = get_group_id(0) * 64;
  const int first_column = get_group_id(1) * 64;
  float4 accumulators[8];
  for (int tile = 0; tile < 8; ++tile) {
    accumulators[tile] = (float4)(0.0f);
  }
  for (int first = 0; first < inner; first += 32) {
    for (int index = id * 2; index < 64 * 32; index += 256) {
      const int row = index / 32;
      const int d = first + index % 32;
      const int offset = row * 40 + index % 32;
      const ushort2 left = first_row + row < patches
                               ? OmniLoadScalarPair(OMNI_SCALAR(input),
                                                    (ulong)(first_row + row) * inner + d, inner - d)
                               : (ushort2)(0);
      const ulong weight_index = (ulong)(first_column + row) * inner + d;
      const ushort2 first_right =
          first_column + row < output_channels
              ? OmniLoadScalarPair(OMNI_SCALAR(first_weight), weight_index, inner - d)
              : (ushort2)(0);
      const ushort2 second_right =
          first_column + row < output_channels
              ? OmniLoadScalarPair(OMNI_SCALAR(second_weight), weight_index, inner - d)
              : (ushort2)(0);
      vstore2(left, 0, inputs + offset);
      vstore2(first_right, 0, first_weights + offset);
      vstore2(second_right, 0, second_weights + offset);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (int d = 0; d < 32; d += 16) {
      const uint4 a = OmniTensorLeft(inputs, warp, lane, d);
#pragma unroll
      for (int tile = 0; tile < 8; ++tile) {
        accumulators[tile] =
            OmniTensorMma(a, OmniTensorRight(first_weights, tile, lane, d), accumulators[tile]);
        accumulators[tile] =
            OmniTensorMma(a, OmniTensorRight(second_weights, tile, lane, d), accumulators[tile]);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#pragma unroll
  for (int tile = 0; tile < 8; ++tile) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int row = first_row + warp * 16 + lane / 4 + item / 2 * 8;
      const int column = first_column + tile * 8 + (lane % 4) * 2 + item % 2;
      if (row < patches && column < output_channels) {
        WriteScalar(OMNI_SCALAR(output), (ulong)row * output_channels + column,
                    accumulators[tile][item] + ReadScalar(OMNI_SCALAR(bias), column));
      }
    }
  }
}
#endif

// One 256-work-item group per patch/output-channel. All remaining vision
// entries use flat element counts, rounded to a local size of 256.
__kernel void PairedPatchEmbedding(__global uchar *input_buffer, ulong input_offset,
                                   __global uchar *first_weight_buffer, ulong first_weight_offset,
                                   __global uchar *second_weight_buffer, ulong second_weight_offset,
                                   __global uchar *bias_buffer, ulong bias_offset, int height,
                                   int width, int channels, int output_channels, int spatial_patch,
                                   __global uchar *output_buffer, ulong output_offset) {
  const ulong output_index = get_group_id(0);
  const int patches = (height / spatial_patch) * (width / spatial_patch);
  if (output_index >= (ulong)patches * output_channels) {
    return;
  }
  const int patch = output_index / output_channels;
  const int channel = output_index % output_channels;
  const int inner = channels * spatial_patch * spatial_patch;
  const ulong weight_first = (ulong)channel * inner;
  const ulong input_first = (ulong)patch * inner;
  float sum = 0.0f;
  for (int d = get_local_id(0); d < inner; d += 256) {
    const float combined = ReadScalar(OMNI_SCALAR(first_weight), weight_first + d) +
                           ReadScalar(OMNI_SCALAR(second_weight), weight_first + d);
    sum = fma(ReadScalar(OMNI_SCALAR(input), input_first + d), combined, sum);
  }
  __local float scratch[256];
  sum = OmniBlockSum(sum, scratch);
  if (get_local_id(0) == 0) {
    WriteScalar(OMNI_SCALAR(output), output_index, sum + ReadScalar(OMNI_SCALAR(bias), channel));
  }
}

__kernel void GridPositionEmbedding(__global uchar *input_buffer, ulong input_offset,
                                    __global uchar *position_weight_buffer,
                                    ulong position_weight_offset, int grid_height, int grid_width,
                                    int position_grid_size, int hidden_size, int merge_size,
                                    __global uchar *output_buffer, ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index >= (ulong)grid_height * grid_width * hidden_size) {
    return;
  }
  const int dimension = index % hidden_size;
  const int patch = OmniMergedPatch(index / hidden_size, grid_width, merge_size);
  const int row = patch / grid_width;
  const int column = patch % grid_width;
  const float source_y =
      grid_height == 1 ? 0.0f
                       : (float)row * (float)(position_grid_size - 1) / (float)(grid_height - 1);
  const float source_x =
      grid_width == 1 ? 0.0f
                      : (float)column * (float)(position_grid_size - 1) / (float)(grid_width - 1);
  const int y0 = (int)floor(source_y);
  const int x0 = (int)floor(source_x);
  const int y1 = min(y0 + 1, position_grid_size - 1);
  const int x1 = min(x0 + 1, position_grid_size - 1);
  const float fy = source_y - (float)y0;
  const float fx = source_x - (float)x0;
  const ulong i00 = ((ulong)y0 * position_grid_size + x0) * hidden_size + dimension;
  const ulong i01 = ((ulong)y0 * position_grid_size + x1) * hidden_size + dimension;
  const ulong i10 = ((ulong)y1 * position_grid_size + x0) * hidden_size + dimension;
  const ulong i11 = ((ulong)y1 * position_grid_size + x1) * hidden_size + dimension;
  const float p00 = ReadScalar(OMNI_SCALAR(position_weight), i00);
  const float p01 = ReadScalar(OMNI_SCALAR(position_weight), i01);
  const float p10 = ReadScalar(OMNI_SCALAR(position_weight), i10);
  const float p11 = ReadScalar(OMNI_SCALAR(position_weight), i11);
  const float top = fma(p01 - p00, fx, p00);
  const float bottom = fma(p11 - p10, fx, p10);
  const float position = RoundScalar(fma(bottom - top, fy, top));
  WriteScalar(OMNI_SCALAR(output), index,
              ReadScalar(OMNI_SCALAR(input), (ulong)patch * hidden_size + dimension) + position);
}

__kernel void Bias(__global uchar *input_buffer, ulong input_offset, __global uchar *bias_buffer,
                   ulong bias_offset, ulong element_count, int row_size,
                   __global uchar *output_buffer, ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < element_count) {
    WriteScalar(OMNI_SCALAR(output), index,
                ReadScalar(OMNI_SCALAR(input), index) +
                    ReadScalar(OMNI_SCALAR(bias), index % row_size));
  }
}

__kernel void BiasResidual(__global uchar *input_buffer, ulong input_offset,
                           __global uchar *bias_buffer, ulong bias_offset,
                           __global uchar *residual_buffer, ulong residual_offset,
                           ulong element_count, int row_size, __global uchar *output_buffer,
                           ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < element_count) {
    const float biased = RoundScalar(ReadScalar(OMNI_SCALAR(input), index) +
                                     ReadScalar(OMNI_SCALAR(bias), index % row_size));
    WriteScalar(OMNI_SCALAR(output), index, biased + ReadScalar(OMNI_SCALAR(residual), index));
  }
}

__kernel void GeluBias(__global uchar *input_buffer, ulong input_offset,
                       __global uchar *bias_buffer, ulong bias_offset, ulong element_count,
                       int row_size, int is_approximate, __global uchar *output_buffer,
                       ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < element_count) {
    const float biased = RoundScalar(ReadScalar(OMNI_SCALAR(input), index) +
                                     ReadScalar(OMNI_SCALAR(bias), index % row_size));
    const float cubic = ((0.044715f * biased) * biased) * biased;
    const float value = is_approximate
                            ? 0.5f * biased * (1.0f + tanh(0.7978845608028654f * (biased + cubic)))
                            : 0.5f * biased * (1.0f + erf(biased * 0.7071067811865475f));
    WriteScalar(OMNI_SCALAR(output), index, value);
  }
}

__kernel void GridQkvRope(__global uchar *qkv_buffer, ulong qkv_offset, int token_count,
                          int grid_height, int grid_width, int merge_size, int head_count,
                          int head_size, __global uchar *query_buffer, ulong query_offset,
                          __global uchar *key_buffer, ulong key_offset,
                          __global uchar *value_buffer, ulong value_offset) {
  const ulong index = get_global_id(0);
  const int hidden_size = head_count * head_size;
  if (index >= (ulong)token_count * hidden_size) {
    return;
  }
  const int dimension = index % head_size;
  const int token = index / hidden_size;
  const int head = (index / head_size) % head_count;
  const int half_size = head_size / 2;
  const int quarter_size = head_size / 4;
  const int paired = dimension < half_size ? dimension + half_size : dimension - half_size;
  const int frequency_dimension = dimension % half_size;
  const int frequency_index = frequency_dimension % quarter_size;
  const int patch = OmniMergedPatch(token, grid_width, merge_size);
  const int position = frequency_dimension < quarter_size ? patch / grid_width : patch % grid_width;
  const float exponent = -2.0f * (float)frequency_index / (float)half_size;
  const float angle = (float)position * pow(10000.0f, exponent);
  const float cosine = cos(angle);
  const float sine = sin(angle);
  const float sign = dimension < half_size ? -1.0f : 1.0f;
  const ulong first = (ulong)token * 3 * hidden_size + head * head_size;
  const float rotated_query = fma(ReadScalar(OMNI_SCALAR(qkv), first + dimension), cosine,
                                  sign * ReadScalar(OMNI_SCALAR(qkv), first + paired) * sine);
  const float rotated_key =
      fma(ReadScalar(OMNI_SCALAR(qkv), first + hidden_size + dimension), cosine,
          sign * ReadScalar(OMNI_SCALAR(qkv), first + hidden_size + paired) * sine);
  WriteScalar(OMNI_SCALAR(query), index, rotated_query);
  WriteScalar(OMNI_SCALAR(key), index, rotated_key);
  OMNI_SCALAR(value)[index] = OMNI_SCALAR(qkv)[first + 2 * hidden_size + dimension];
}
