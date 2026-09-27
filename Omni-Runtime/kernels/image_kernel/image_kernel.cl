// Flat pixel_count work-items rounded to local 256, including the resize path.
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double omni_image_real;
#elif defined(__opencl_c_fp64)
typedef double omni_image_real;
#else
// Devices without FP64 retain the same filter, but coefficient ties may differ.
typedef float omni_image_real;
#endif

inline omni_image_real OmniCubic(omni_image_real distance) {
  const omni_image_real x = fabs(distance);
  if (x < (omni_image_real)1) {
    return (((omni_image_real)1.5 * x - (omni_image_real)2.5) * x) * x + (omni_image_real)1;
  }
  return x < (omni_image_real)2
             ? (((omni_image_real)-0.5 * x + (omni_image_real)2.5) * x - (omni_image_real)4) * x +
                   (omni_image_real)2
             : (omni_image_real)0;
}

inline int OmniQuantizedWeight(omni_image_real weight, int precision) {
  const omni_image_real scaled = weight * (omni_image_real)(1 << precision);
  return (int)(scaled < (omni_image_real)0 ? ceil(scaled - (omni_image_real)0.5)
                                           : floor(scaled + (omni_image_real)0.5));
}

inline int OmniHorizontalPixel(__global const uchar *source, int row, int column, int channel,
                               int input_width, int output_width, int precision) {
  if (input_width == output_width) {
    return source[((ulong)row * input_width + column) * 3 + channel];
  }
  const omni_image_real scale = (omni_image_real)input_width / (omni_image_real)output_width;
  const omni_image_real filter_scale = fmax(scale, (omni_image_real)1);
  const omni_image_real center = ((omni_image_real)column + (omni_image_real)0.5) * scale;
  const int begin =
      max(0, (int)(center - (omni_image_real)2 * filter_scale + (omni_image_real)0.5));
  const int end =
      min(input_width, (int)(center + (omni_image_real)2 * filter_scale + (omni_image_real)0.5));
  omni_image_real sum = (omni_image_real)0;
  for (int i = begin; i < end; ++i) {
    sum += OmniCubic(((omni_image_real)i - center + (omni_image_real)0.5) / filter_scale);
  }
  int value = 1 << (precision - 1);
  for (int i = begin; i < end; ++i) {
    const int weight = OmniQuantizedWeight(
        OmniCubic(((omni_image_real)i - center + (omni_image_real)0.5) / filter_scale) / sum,
        precision);
    value += weight * source[((ulong)row * input_width + i) * 3 + channel];
  }
  return clamp(value >> precision, 0, 255);
}

__kernel void ImagePatchPreprocess(ulong pixel_count, __global uchar *source_buffer,
                                   ulong source_offset, int source_width_pixels,
                                   int source_height_pixels, int patch_size,
                                   __global uchar *destination_buffer, ulong destination_offset,
                                   int input_width_pixels, int input_height_pixels, int x_precision,
                                   int y_precision) {
  const ulong index = get_global_id(0);
  const int resize = input_width_pixels > 0 && input_height_pixels > 0;
  if (index >= pixel_count) {
    return;
  }
  const int row = index / source_width_pixels;
  const int column = index % source_width_pixels;
  if (row >= source_height_pixels) {
    return;
  }
  __global const uchar *source = source_buffer + source_offset;
  const int area = patch_size * patch_size;
  const int patch = (row / patch_size) * (source_width_pixels / patch_size) + column / patch_size;
  const int local_pixel = (row % patch_size) * patch_size + column % patch_size;
  const ulong destination = (ulong)patch * 3 * area + local_pixel;
  if (!resize) {
    for (int channel = 0; channel < 3; ++channel) {
      const float value = (float)source[index * 3 + channel] / 127.5f - 1.0f;
      WriteScalar(OMNI_SCALAR(destination), destination + channel * area, value);
    }
    return;
  }
  const omni_image_real scale =
      (omni_image_real)input_height_pixels / (omni_image_real)source_height_pixels;
  const omni_image_real filter_scale = fmax(scale, (omni_image_real)1);
  const omni_image_real center = ((omni_image_real)row + (omni_image_real)0.5) * scale;
  const int begin =
      max(0, (int)(center - (omni_image_real)2 * filter_scale + (omni_image_real)0.5));
  const int end = min(input_height_pixels,
                      (int)(center + (omni_image_real)2 * filter_scale + (omni_image_real)0.5));
  omni_image_real weight_sum = (omni_image_real)0;
  for (int y = begin; y < end; ++y) {
    weight_sum += OmniCubic(((omni_image_real)y - center + (omni_image_real)0.5) / filter_scale);
  }
  for (int channel = 0; channel < 3; ++channel) {
    int sum = 1 << (y_precision - 1);
    for (int y = begin; y < end; ++y) {
      const int weight = OmniQuantizedWeight(
          OmniCubic(((omni_image_real)y - center + (omni_image_real)0.5) / filter_scale) /
              weight_sum,
          y_precision);
      sum += weight * OmniHorizontalPixel(source, y, column, channel, input_width_pixels,
                                          source_width_pixels, x_precision);
    }
    const int value = clamp(sum >> y_precision, 0, 255);
    WriteScalar(OMNI_SCALAR(destination), destination + channel * area,
                ((float)value - 127.5f) / 127.5f);
  }
}
