#pragma once

#include <cstdint>
#include <variant>

namespace omni_runtime::backend {

struct Embedding final {
  const void *const indices;
  const void *const weight;
  int32_t token_count;
  int32_t hidden_size;
  void *const output;
};

struct RmsNorm final {
  const void *const input;
  const void *const weight;
  int32_t rows;
  int32_t hidden_size;
  float epsilon;
  void *const output;
};

struct LayerNorm final {
  const void *const input;
  const void *const weight;
  const void *const bias;
  int32_t rows;
  int32_t hidden_size;
  float epsilon;
  void *const output;
};

struct MultiAxisRope final {
  const void *const query;
  const void *const key;
  const void *const position_ids;
  int32_t tokens;
  int32_t query_heads;
  int32_t key_value_heads;
  int32_t head_dim;
  int32_t temporal_section;
  int32_t height_section;
  int32_t width_section;
  float frequency_base;
  void *const query_output;
  void *const key_output;
};

struct StoreKvCache final {
  const void *const key;
  const void *const value;
  int32_t token_count;
  int32_t key_value_size;
  int32_t start_position;
  int32_t cache_capacity;
  void *const key_cache;
  void *const value_cache;
};

struct Attention final {
  const void *const query;
  const void *const key_cache;
  const void *const value_cache;
  int32_t query_token_count;
  int32_t query_head_count;
  int32_t key_value_head_count;
  int32_t cached_token_count;
  int32_t start_position;
  int32_t head_dim;
  bool is_causal;
  void *const output;
  void *decode_partials = nullptr;
};

struct Add final {
  const void *const lhs;
  const void *const rhs;
  uint64_t element_count;
  void *const output;
};

struct SwiGLU final {
  const void *const gate;
  const void *const up;
  uint64_t element_count;
  void *const output;
};

struct Argmax final {
  const void *const logits;
  int32_t rows;
  int32_t columns;
  void *const output;
};

struct ImagePatchPreprocess final {
  uint64_t pixel_count;
  const void *const source;
  int32_t source_width_pixels;
  int32_t source_height_pixels;
  int32_t patch_size;
  void *const destination;
  int32_t input_width_pixels = 0;
  int32_t input_height_pixels = 0;
};

struct PairedPatchEmbedding final {
  const void *const input;
  const void *const first_weight;
  const void *const second_weight;
  const void *const bias;
  int32_t height;
  int32_t width;
  int32_t channels;
  int32_t output_channels;
  int32_t spatial_patch;
  void *const output;
};

struct GridPositionEmbedding final {
  const void *const input;
  const void *const position_weight;
  int32_t grid_height;
  int32_t grid_width;
  int32_t position_grid_size;
  int32_t hidden_size;
  int32_t merge_size;
  void *const output;
};

struct Bias final {
  const void *const input;
  const void *const bias;
  uint64_t element_count;
  int32_t row_size;
  void *const output;
};

struct BiasResidual final {
  const void *const input;
  const void *const bias;
  const void *const residual;
  uint64_t element_count;
  int32_t row_size;
  void *const output;
};

struct GeluBias final {
  const void *const input;
  const void *const bias;
  uint64_t element_count;
  int32_t row_size;
  bool is_approximate = true;
  void *const output;
};

struct GridQkvRope final {
  const void *const qkv;
  int32_t token_count;
  int32_t grid_height;
  int32_t grid_width;
  int32_t merge_size;
  int32_t head_count;
  int32_t head_size;
  void *const query;
  void *const key;
  void *const value;
};

using ComputeOperation =
    std::variant<Embedding, RmsNorm, LayerNorm, MultiAxisRope, StoreKvCache, Attention, Add, SwiGLU,
                 Argmax, ImagePatchPreprocess, PairedPatchEmbedding, GridPositionEmbedding, Bias,
                 BiasResidual, GeluBias, GridQkvRope>;

} // namespace omni_runtime::backend
