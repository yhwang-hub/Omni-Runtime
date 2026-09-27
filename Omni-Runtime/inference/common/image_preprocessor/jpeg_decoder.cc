#include "Omni-Runtime/inference/common/image_preprocessor/jpeg_decoder.h"

#include <limits>
#include <memory>
#include <utility>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {

JpegDecoder::State::~State() {
  jpeg_destroy_decompress(&decoder);
}

bool JpegDecoder::Load(FILE *const source, std::vector<uint8_t> *const pixels,
                       int32_t *const width_pixels, int32_t *const height_pixels) {
  OMNI_RETURN_VAL_IF_LOG(source == nullptr || pixels == nullptr || width_pixels == nullptr ||
                             height_pixels == nullptr,
                         false, ERROR, "JPEG source or output is null");
  // Keep libjpeg's mutated state off the setjmp stack and construct owners before setjmp.
  const auto state = std::make_unique<State>();
  std::vector<uint8_t> &decoded_pixels = state->pixels;
  jpeg_decompress_struct &decoder = state->decoder;
  decoder.err = jpeg_std_error(&state->error.manager);
  state->error.manager.error_exit = Fail;
  if (setjmp(state->error.jump) != 0) {
    LOG_ERROR("JPEG decode failed: " << state->error.message);
    return false;
  }
  jpeg_create_decompress(&decoder);
  jpeg_stdio_src(&decoder, source);
  jpeg_read_header(&decoder, TRUE);
  decoder.out_color_space = JCS_RGB;
  jpeg_start_decompress(&decoder);
  const uint64_t size_bytes =
      static_cast<uint64_t>(decoder.output_width) * decoder.output_height * 3U;
  if (decoder.output_width == 0U || decoder.output_height == 0U ||
      decoder.output_width > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      decoder.output_height > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      size_bytes > decoded_pixels.max_size()) {
    LOG_ERROR("JPEG dimensions are invalid");
    return false;
  }
  decoded_pixels.resize(static_cast<size_t>(size_bytes));
  while (decoder.output_scanline < decoder.output_height) {
    JSAMPROW rows[] = {decoded_pixels.data() +
                       static_cast<size_t>(decoder.output_scanline) * decoder.output_width * 3U};
    jpeg_read_scanlines(&decoder, rows, 1U);
  }
  const int32_t decoded_width = static_cast<int32_t>(decoder.output_width);
  const int32_t decoded_height = static_cast<int32_t>(decoder.output_height);
  jpeg_finish_decompress(&decoder);
  *pixels = std::move(decoded_pixels);
  *width_pixels = decoded_width;
  *height_pixels = decoded_height;
  return true;
}

void JpegDecoder::Fail(j_common_ptr const decoder) {
  auto *const error = reinterpret_cast<Error *>(decoder->err);
  decoder->err->format_message(decoder, error->message);
  longjmp(error->jump, 1);
}

} // namespace inference
} // namespace omni_runtime
