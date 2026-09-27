#pragma once

#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <jpeglib.h>

namespace omni_runtime {
namespace inference {

class JpegDecoder final {
public:
  static bool Load(FILE *const source, std::vector<uint8_t> *const pixels,
                   int32_t *const width_pixels, int32_t *const height_pixels);

private:
  struct Error final {
    jpeg_error_mgr manager = {};
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX] = {};
  };

  struct State final {
    ~State();

    jpeg_decompress_struct decoder = {};
    Error error;
    std::vector<uint8_t> pixels;
  };

  static void Fail(j_common_ptr const decoder);
};

} // namespace inference
} // namespace omni_runtime
