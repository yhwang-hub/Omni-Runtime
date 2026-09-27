__kernel void PatchEmbedding(__global uchar *input_buffer, ulong input_offset,
                             __global uchar *weight_buffer, ulong weight_offset,
                             __global uchar *bias_buffer, ulong bias_offset, int frames, int height,
                             int width, int channels, int output_channels, int temporal_patch,
                             int spatial_patch, __global uchar *output_buffer,
                             ulong output_offset) {
  const ulong output_index = get_group_id(0);
  const int output_height = height / spatial_patch;
  const int output_width = width / spatial_patch;
  const int patch = output_index / output_channels;
  const int output_channel = output_index % output_channels;
  const int inner = channels * temporal_patch * spatial_patch * spatial_patch;
  float sum = 0.0f;
  for (int d = get_local_id(0); d < inner; d += 256) {
    const int x = (patch % output_width) * spatial_patch + d % spatial_patch;
    const int y = ((patch / output_width) % output_height) * spatial_patch +
                  (d / spatial_patch) % spatial_patch;
    const int t = patch / (output_height * output_width) * temporal_patch +
                  (d / (spatial_patch * spatial_patch)) % temporal_patch;
    const int channel = d / (temporal_patch * spatial_patch * spatial_patch);
    const ulong input_index = (((ulong)channel * frames + t) * height + y) * width + x;
    sum = fma(ReadScalar(OMNI_SCALAR(input), input_index),
              ReadScalar(OMNI_SCALAR(weight), (ulong)output_channel * inner + d), sum);
  }
  __local float scratch[256];
  sum = OmniBlockSum(sum, scratch);
  if (get_local_id(0) == 0) {
    WriteScalar(OMNI_SCALAR(output), output_index,
                sum + ReadScalar(OMNI_SCALAR(bias), output_channel));
  }
}

__kernel void PatchMerge(__global uchar *input_buffer, ulong input_offset, int token_count,
                         int grid_width, int hidden_size, int merge_size,
                         __global uchar *output_buffer, ulong output_offset) {
  const ulong index = get_global_id(0);
  if (index < (ulong)token_count * hidden_size) {
    OMNI_SCALAR(output)[index] = OMNI_SCALAR(input)[index];
  }
}
