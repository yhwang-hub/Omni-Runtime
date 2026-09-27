#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace omni_runtime {
namespace inference {

// Receives complete UTF-8 text increments; false cancels generation successfully.
using TextStreamCallback = std::function<bool(const std::string &)>;

enum class ModelBackend : uint8_t {
  kAuto = 0,
  kCpu = 1,
  kGpu = 2,
};

enum class ModelWeightType : uint8_t {
  kAuto = 0,
  kFloat32 = 1,
  kFloat16 = 2,
  kBFloat16 = 3,
  kInt8 = 4,
};

struct GenerationRequest final {
  std::string prompt;
  std::string image_path;
  int32_t max_new_tokens = 0;
  TextStreamCallback stream_callback;
};

struct ImageSize final {
  int32_t width_pixels = 0;
  int32_t height_pixels = 0;
};

struct GenerationResult final {
  std::vector<int32_t> prompt_token_ids;
  std::vector<int32_t> generated_token_ids;
  std::string text;
  std::string backend_name;
  ImageSize input_image_size;
  ImageSize resized_image_size;
  int32_t prefill_token_count = 0;
  int32_t vision_token_count = 0;
  int32_t decode_token_count = 0;
  double prefill_time_ms = 0.0;
  double vision_time_ms = 0.0;
  double decode_time_ms = 0.0;
  size_t gpu_memory_bytes = 0U;
  bool is_cancelled = false;

  double prefill_tokens_per_second() const {
    return prefill_time_ms > 0.0 ? 1000.0 * prefill_token_count / prefill_time_ms : 0.0;
  }

  double decode_tokens_per_second() const {
    return decode_time_ms > 0.0 ? 1000.0 * decode_token_count / decode_time_ms : 0.0;
  }
};

const char *ModelBackendName(const ModelBackend backend);
const char *ModelWeightTypeName(const ModelWeightType weight_type);

} // namespace inference
} // namespace omni_runtime
