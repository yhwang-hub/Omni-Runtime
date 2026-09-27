#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_vision_encoder_executor_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <csetjmp>
#include <cstdio>

#include <jpeglib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "Omni-Runtime/inference/tokenizer/tokenizer.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_runtime.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {
namespace {

using memory::BufferPtr;

struct VisionGrid final {
  int32_t height = 0;
  int32_t width = 0;
  int32_t token_count = 0;
  int32_t merged_token_count = 0;
};

using VisionWorkspaces = Qwen3VLVisionWorkspaces;

class JpegDecoder final {
public:
  static bool Load(const std::string &path, std::vector<uint8_t> *const pixels,
                   int32_t *const width, int32_t *const height) {
    OMNI_RETURN_VAL_IF(pixels == nullptr || width == nullptr || height == nullptr, false);
    FILE *const file = std::fopen(path.c_str(), "rb");
    OMNI_RETURN_VAL_IF(file == nullptr, false);
    jpeg_decompress_struct decoder = {};
    Error error;
    decoder.err = jpeg_std_error(&error.manager);
    error.manager.error_exit = Fail;
    if (setjmp(error.jump) != 0) {
      jpeg_destroy_decompress(&decoder);
      std::fclose(file);
      LOG_ERROR("JPEG decode failed: " << error.message);
      return false;
    }
    jpeg_create_decompress(&decoder);
    jpeg_stdio_src(&decoder, file);
    jpeg_read_header(&decoder, TRUE);
    decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decoder);
    *width = decoder.output_width;
    *height = decoder.output_height;
    pixels->resize(static_cast<size_t>(*width) * *height * 3U);
    while (decoder.output_scanline < decoder.output_height) {
      JSAMPROW rows[] = {pixels->data() +
                         static_cast<size_t>(decoder.output_scanline) * *width * 3U};
      jpeg_read_scanlines(&decoder, rows, 1U);
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    std::fclose(file);
    return true;
  }

private:
  struct Error final {
    jpeg_error_mgr manager;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX] = {};
  };

  static void Fail(j_common_ptr decoder) {
    auto *const error = reinterpret_cast<Error *>(decoder->err);
    decoder->err->format_message(decoder, error->message);
    longjmp(error->jump, 1);
  }
};

bool LoadImage(const std::string &image_path, std::vector<std::uint8_t> *const pixels,
               int32_t *const width, int32_t *const height) {
  OMNI_RETURN_VAL_IF(pixels == nullptr || width == nullptr || height == nullptr, false);
  FILE *const file = std::fopen(image_path.c_str(), "rb");
  OMNI_RETURN_VAL_IF(file == nullptr, false);
  const int first = std::fgetc(file);
  const int second = std::fgetc(file);
  std::fclose(file);
  if (first == 0xFF && second == 0xD8) {
    return JpegDecoder::Load(image_path, pixels, width, height);
  }
  int source_width = 0;
  int source_height = 0;
  int channel_count = 0;
  std::uint8_t *const source =
      stbi_load(image_path.c_str(), &source_width, &source_height, &channel_count, 3);
  OMNI_RETURN_VAL_IF(source == nullptr || source_width <= 0 || source_height <= 0, false);
  pixels->assign(source, source + static_cast<size_t>(source_width) * source_height * 3U);
  stbi_image_free(source);
  *width = source_width;
  *height = source_height;
  return true;
}

bool MakeGrid(const int32_t source_width, const int32_t source_height,
              const Qwen3VLVisionConfig &config, VisionGrid *const grid) {
  OMNI_RETURN_VAL_IF(grid == nullptr || source_width <= 0 || source_height <= 0, false);
  const auto aligned_size = [&config](const int32_t size) {
    return std::max(config.patch_alignment(), (size + config.patch_alignment() - 1) /
                                                  config.patch_alignment() *
                                                  config.patch_alignment());
  };
  const int32_t width = aligned_size(source_width);
  const int32_t height = aligned_size(source_height);
  grid->height = height / config.patch_size;
  grid->width = width / config.patch_size;
  grid->token_count = grid->height * grid->width;
  grid->merged_token_count = grid->token_count / (config.merge_size * config.merge_size);
  return grid->merged_token_count > 0;
}

bool LaunchPatchEmbedding(Qwen3VLRuntime *const runtime, const VisionGrid &grid,
                          const int32_t source_width, const int32_t source_height,
                          const VisionWorkspaces &workspaces) {
  const Qwen3VLVisionConfig &config = runtime->vision_config;
  const uint64_t pixel_count =
      static_cast<uint64_t>(grid.width) * config.patch_size * grid.height * config.patch_size;
  return runtime->backend->Compute(
             RuntimeScalarType(*runtime),
             backend::ImagePatchPreprocess{.pixel_count = pixel_count,
                                           .source = workspaces.source->device_address(),
                                           .source_width_pixels = grid.width * config.patch_size,
                                           .source_height_pixels = grid.height * config.patch_size,
                                           .patch_size = config.patch_size,
                                           .destination = workspaces.patches->device_address(),
                                           .input_width_pixels = source_width,
                                           .input_height_pixels = source_height}) &&
         runtime->backend->Compute(RuntimeScalarType(*runtime),
                                   backend::PairedPatchEmbedding{
                                       .input = workspaces.patches->device_address(),
                                       .first_weight = Weight(*runtime, "v.patch_embd.weight"),
                                       .second_weight = Weight(*runtime, "v.patch_embd.weight.1"),
                                       .bias = Weight(*runtime, "v.patch_embd.bias"),
                                       .height = grid.height * config.patch_size,
                                       .width = grid.width * config.patch_size,
                                       .channels = 3,
                                       .output_channels = config.hidden_size,
                                       .spatial_patch = config.patch_size,
                                       .output = workspaces.patch_embedding->device_address()}) &&
         runtime->backend->Compute(
             RuntimeScalarType(*runtime),
             backend::GridPositionEmbedding{.input = workspaces.patch_embedding->device_address(),
                                            .position_weight =
                                                Weight(*runtime, "v.position_embd.weight"),
                                            .grid_height = grid.height,
                                            .grid_width = grid.width,
                                            .position_grid_size = 48,
                                            .hidden_size = config.hidden_size,
                                            .merge_size = config.merge_size,
                                            .output = workspaces.hidden->device_address()});
}

bool LaunchVisionLayerNorm(Qwen3VLRuntime *const runtime, const void *const input,
                           const std::string &weight_name, const std::string &bias_name,
                           const int32_t rows, const int32_t hidden_size, void *const output) {
  return runtime->backend->Compute(
      RuntimeScalarType(*runtime),
      backend::LayerNorm{.input = input,
                         .weight = Weight(*runtime, weight_name),
                         .bias = Weight(*runtime, bias_name),
                         .rows = rows,
                         .hidden_size = hidden_size,
                         .epsilon = runtime->vision_config.layer_norm_epsilon,
                         .output = output});
}

bool LaunchBias(Qwen3VLRuntime *const runtime, const void *const input,
                const std::string &bias_name, const uint64_t element_count, const int32_t row_size,
                void *const output) {
  return runtime->backend->Compute(RuntimeScalarType(*runtime),
                                   backend::Bias{.input = input,
                                                 .bias = Weight(*runtime, bias_name),
                                                 .element_count = element_count,
                                                 .row_size = row_size,
                                                 .output = output});
}

bool LaunchBiasResidual(Qwen3VLRuntime *const runtime, const void *const input,
                        const std::string &bias_name, const void *const residual,
                        const uint64_t element_count, const int32_t row_size, void *const output) {
  return runtime->backend->Compute(RuntimeScalarType(*runtime),
                                   backend::BiasResidual{.input = input,
                                                         .bias = Weight(*runtime, bias_name),
                                                         .residual = residual,
                                                         .element_count = element_count,
                                                         .row_size = row_size,
                                                         .output = output});
}

bool LaunchGeluBias(Qwen3VLRuntime *const runtime, const void *const input,
                    const std::string &bias_name, const uint64_t element_count,
                    const int32_t row_size, const bool is_approximate, void *const output) {
  return runtime->backend->Compute(RuntimeScalarType(*runtime),
                                   backend::GeluBias{.input = input,
                                                     .bias = Weight(*runtime, bias_name),
                                                     .element_count = element_count,
                                                     .row_size = row_size,
                                                     .is_approximate = is_approximate,
                                                     .output = output});
}

bool LaunchVisionAttention(Qwen3VLRuntime *const runtime, const std::string &prefix,
                           const VisionGrid &grid, const VisionWorkspaces &workspaces) {
  const Qwen3VLVisionConfig &config = runtime->vision_config;
  const uint64_t hidden_values = static_cast<uint64_t>(grid.token_count) * config.hidden_size;
  return runtime->backend->Compute(
             RuntimeScalarType(*runtime),
             backend::GridQkvRope{.qkv = workspaces.qkv->device_address(),
                                  .token_count = grid.token_count,
                                  .grid_height = grid.height,
                                  .grid_width = grid.width,
                                  .merge_size = config.merge_size,
                                  .head_count = config.head_count,
                                  .head_size = config.head_size(),
                                  .query = workspaces.query->device_address(),
                                  .key = workspaces.key->device_address(),
                                  .value = workspaces.value->device_address()}) &&
         runtime->backend->Compute(
             RuntimeScalarType(*runtime),
             backend::Attention{.query = workspaces.query->device_address(),
                                .key_cache = workspaces.key->device_address(),
                                .value_cache = workspaces.value->device_address(),
                                .query_token_count = grid.token_count,
                                .query_head_count = config.head_count,
                                .key_value_head_count = config.head_count,
                                .cached_token_count = grid.token_count,
                                .start_position = 0,
                                .head_dim = config.head_size(),
                                .is_causal = false,
                                .output = workspaces.attention->device_address()}) &&
         LaunchMatmul(runtime, workspaces.attention->device_address(), prefix + "attn_out.weight",
                      grid.token_count, config.hidden_size, config.hidden_size,
                      workspaces.projected->device_address()) &&
         LaunchBiasResidual(runtime, workspaces.projected->device_address(),
                            prefix + "attn_out.bias", workspaces.hidden->device_address(),
                            hidden_values, config.hidden_size,
                            workspaces.residual->device_address());
}

bool LaunchProjection(Qwen3VLRuntime *const runtime, const std::string &prefix,
                      const std::string &first_suffix, const std::string &second_suffix,
                      const VisionGrid &grid, const VisionWorkspaces &workspaces,
                      const void *const merged, void *const output) {
  const Qwen3VLVisionConfig &config = runtime->vision_config;
  const uint64_t intermediate_values =
      static_cast<uint64_t>(grid.merged_token_count) * config.merged_hidden_size();
  return LaunchMatmul(runtime, merged, prefix + first_suffix + ".weight", grid.merged_token_count,
                      config.merged_hidden_size(), config.merged_hidden_size(),
                      workspaces.qkv->device_address()) &&
         LaunchGeluBias(runtime, workspaces.qkv->device_address(), prefix + first_suffix + ".bias",
                        intermediate_values, config.merged_hidden_size(), false,
                        workspaces.qkv->device_address()) &&
         LaunchMatmul(runtime, workspaces.qkv->device_address(), prefix + second_suffix + ".weight",
                      grid.merged_token_count, config.merged_hidden_size(), config.projection_size,
                      output) &&
         LaunchBias(runtime, output, prefix + second_suffix + ".bias",
                    static_cast<uint64_t>(grid.merged_token_count) * config.projection_size,
                    config.projection_size, output);
}

bool RunVisionEncoder(Qwen3VLRuntime *const runtime, const std::string &image_path,
                      Qwen3VLVisionFeatures *const features) {
  OMNI_RETURN_VAL_IF(runtime == nullptr || features == nullptr, false);
  std::vector<std::uint8_t> source_pixels;
  int32_t source_width = 0;
  int32_t source_height = 0;
  VisionGrid grid;
  if (!LoadImage(image_path, &source_pixels, &source_width, &source_height)) {
    LOG_ERROR("Qwen3-VL image load failed: " << image_path);
    return false;
  }
  if (!MakeGrid(source_width, source_height, runtime->vision_config, &grid)) {
    LOG_ERROR("Qwen3-VL image grid failed");
    return false;
  }
  OMNI_RETURN_VAL_IF_LOG(grid.merged_token_count > runtime->config->max_context_tokens(), false,
                         ERROR, "Qwen3-VL image exceeds workspace capacity");
  const VisionWorkspaces &workspaces = runtime->vision_workspaces;
  OMNI_RETURN_VAL_IF(!runtime->compute_timer.Start(), false);
  if (!workspaces.source->CopyFromHost(source_pixels.data(), source_pixels.size())) {
    LOG_ERROR("Qwen3-VL vision image upload failed");
    return false;
  }
  if (!LaunchPatchEmbedding(runtime, grid, source_width, source_height, workspaces)) {
    LOG_ERROR("Qwen3-VL vision patch embedding failed");
    return false;
  }
  for (int32_t layer = 0; layer < runtime->vision_config.layer_count; ++layer) {
    const std::string prefix = "v.blk." + std::to_string(layer) + ".";
    const uint64_t hidden_values =
        static_cast<uint64_t>(grid.token_count) * runtime->vision_config.hidden_size;
    const bool is_layer_successful =
        LaunchVisionLayerNorm(runtime, workspaces.hidden->device_address(), prefix + "ln1.weight",
                              prefix + "ln1.bias", grid.token_count,
                              runtime->vision_config.hidden_size,
                              workspaces.normalized->device_address()) &&
        LaunchMatmul(runtime, workspaces.normalized->device_address(), prefix + "attn_qkv.weight",
                     grid.token_count, runtime->vision_config.hidden_size,
                     runtime->vision_config.hidden_size * 3, workspaces.qkv->device_address()) &&
        LaunchBias(runtime, workspaces.qkv->device_address(), prefix + "attn_qkv.bias",
                   hidden_values * 3U, runtime->vision_config.hidden_size * 3,
                   workspaces.qkv->device_address()) &&
        LaunchVisionAttention(runtime, prefix, grid, workspaces) &&
        LaunchVisionLayerNorm(runtime, workspaces.residual->device_address(), prefix + "ln2.weight",
                              prefix + "ln2.bias", grid.token_count,
                              runtime->vision_config.hidden_size,
                              workspaces.normalized->device_address()) &&
        LaunchMatmul(runtime, workspaces.normalized->device_address(), prefix + "ffn_up.weight",
                     grid.token_count, runtime->vision_config.hidden_size,
                     runtime->vision_config.feed_forward_size,
                     workspaces.feed_forward->device_address()) &&
        LaunchGeluBias(runtime, workspaces.feed_forward->device_address(), prefix + "ffn_up.bias",
                       static_cast<uint64_t>(grid.token_count) *
                           runtime->vision_config.feed_forward_size,
                       runtime->vision_config.feed_forward_size, true,
                       workspaces.feed_forward->device_address()) &&
        LaunchMatmul(runtime, workspaces.feed_forward->device_address(), prefix + "ffn_down.weight",
                     grid.token_count, runtime->vision_config.feed_forward_size,
                     runtime->vision_config.hidden_size, workspaces.projected->device_address()) &&
        LaunchBiasResidual(runtime, workspaces.projected->device_address(),
                           prefix + "ffn_down.bias", workspaces.residual->device_address(),
                           hidden_values, runtime->vision_config.hidden_size,
                           workspaces.hidden->device_address());
    if (!is_layer_successful) {
      LOG_ERROR("Qwen3-VL vision layer failed: " << layer);
      return false;
    }
    if (std::find(runtime->vision_config.deepstack_layers.begin(),
                  runtime->vision_config.deepstack_layers.end(),
                  layer) == runtime->vision_config.deepstack_layers.end()) {
      continue;
    }
    const BufferPtr deepstack = workspaces.deepstack[features->deepstack_embeddings.size()];
    const bool is_deepstack_successful =
        LaunchVisionLayerNorm(runtime, workspaces.hidden->device_address(),
                              "v.deepstack." + std::to_string(layer) + ".norm.weight",
                              "v.deepstack." + std::to_string(layer) + ".norm.bias",
                              grid.merged_token_count, runtime->vision_config.merged_hidden_size(),
                              workspaces.normalized->device_address()) &&
        LaunchProjection(runtime, "v.deepstack." + std::to_string(layer) + ".", "fc1", "fc2", grid,
                         workspaces, workspaces.normalized->device_address(),
                         deepstack->device_address());
    if (!is_deepstack_successful) {
      LOG_ERROR("Qwen3-VL deepstack layer failed: " << layer);
      return false;
    }
    features->deepstack_embeddings.emplace_back(deepstack);
  }

  const bool is_output_successful =
      LaunchVisionLayerNorm(runtime, workspaces.hidden->device_address(), "v.post_ln.weight",
                            "v.post_ln.bias", grid.token_count, runtime->vision_config.hidden_size,
                            workspaces.normalized->device_address()) &&
      LaunchProjection(runtime, "mm.", "0", "2", grid, workspaces,
                       workspaces.normalized->device_address(),
                       workspaces.embedding->device_address());
  if (!is_output_successful) {
    LOG_ERROR("Qwen3-VL vision output projection failed");
    return false;
  }
  features->embedding = workspaces.embedding;
  features->input_image_size = {.width_pixels = source_width, .height_pixels = source_height};
  features->resized_image_size = {.width_pixels = grid.width * runtime->vision_config.patch_size,
                                  .height_pixels = grid.height * runtime->vision_config.patch_size};
  features->token_count = grid.merged_token_count;
  features->merged_grid_height = grid.height / runtime->vision_config.merge_size;
  features->merged_grid_width = grid.width / runtime->vision_config.merge_size;
  return runtime->compute_timer.Stop(&runtime->step_time_ms);
}

} // namespace

Qwen3VLVisionEncoderExecutorGpu::Qwen3VLVisionEncoderExecutorGpu(
    Qwen3VLRuntime *const runtime, Qwen3VLGenerationContext *const context)
    : runtime_(runtime), context_(context) {
}

bool Qwen3VLVisionEncoderExecutorGpu::Initialize() {
  OMNI_RETURN_VAL_IF(runtime_ == nullptr, SetError("Qwen3-VL vision runtime is null"));
  return true;
}

graph::Status Qwen3VLVisionEncoderExecutorGpu::Run() {
  error_message_.clear();
  OMNI_RETURN_VAL_IF_LOG(context_ == nullptr || context_->request == nullptr ||
                             context_->result == nullptr,
                         graph::InvalidArgumentStatus("Qwen3-VL vision context is null"), ERROR,
                         "Qwen3-VL vision context is null");
  context_->vision_features = {};
  if (context_->request->image_path.empty()) {
    return graph::Status();
  }
  const bool is_successful =
      RunVisionEncoder(runtime_, context_->request->image_path, &context_->vision_features);
  OMNI_RETURN_VAL_IF_LOG(!is_successful,
                         graph::InternalStatus(error_message_.empty()
                                                   ? "Qwen3-VL vision encoder failed"
                                                   : error_message_),
                         ERROR, error_message_);
  context_->result->vision_time_ms = runtime_->step_time_ms;
  context_->result->input_image_size = context_->vision_features.input_image_size;
  context_->result->resized_image_size = context_->vision_features.resized_image_size;
  return graph::Status();
}

bool Qwen3VLVisionEncoderExecutorGpu::SetError(const std::string &message) {
  error_message_ = message;
  if (context_ != nullptr) {
    context_->error_message = message;
  }
  return false;
}

const std::string &Qwen3VLVisionEncoderExecutorGpu::error_message() const {
  return error_message_;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
