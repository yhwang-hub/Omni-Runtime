#pragma once

#include <cstdint>
#include <variant>
#include <vector>

#include "Omni-Runtime/backend/backend_define.h"

namespace omni_runtime::backend {

struct AdaRmsNorm final {
  DeviceView input;
  DeviceView scale;
  DeviceView shift;
  int32_t rows;
  int32_t hidden_size;
  float epsilon;
  DeviceView output;
};

struct FlowStep final {
  DeviceView state;
  DeviceView velocity;
  float dt;
  uint64_t element_count;
  DeviceView output;
};

struct GatedAdd final {
  DeviceView lhs;
  DeviceView rhs;
  DeviceView gate;
  uint64_t element_count;
  int32_t hidden_size;
  DeviceView output;
};

struct SplitGatedQuery final {
  DeviceView query_gate;
  int32_t token_count;
  int32_t head_count;
  int32_t head_size;
  DeviceView query;
  DeviceView gate;
};

struct FusedFfn final {
  DeviceView input;
  DeviceView fc1_weight;
  DeviceView fc1_bias;
  DeviceView fc2_weight;
  DeviceView fc2_bias;
  int32_t rows;
  int32_t input_size;
  int32_t intermediate_size;
  int32_t output_size;
  DeviceView output;
};

// Unlike Attention, the standalone MHA API stores [head, token, dimension].
struct MultiHeadAttention final {
  DeviceView query;
  DeviceView key;
  DeviceView value;
  int32_t heads;
  int32_t tokens;
  int32_t head_dim;
  bool is_causal;
  DeviceView output;
};

struct Linear final {
  DeviceView input;
  DeviceView weight;
  int32_t rows;
  int32_t inner;
  int32_t columns;
  DeviceView output;
};

// A fork/join of two or three projections sharing the same input and shape.
struct ParallelLinear final {
  std::vector<Linear> projections;
};

bool ValidateParallelLinear(const ScalarType type, const BackendKind kind,
                            const ParallelLinear &operation);

struct Embedding final {
  DeviceView indices;
  DeviceView weight;
  int32_t token_count;
  int32_t hidden_size;
  DeviceView output;
};

struct RmsNorm final {
  DeviceView input;
  DeviceView weight;
  int32_t rows;
  int32_t hidden_size;
  float epsilon;
  DeviceView output;
};

struct LayerNorm final {
  DeviceView input;
  DeviceView weight;
  DeviceView bias;
  int32_t rows;
  int32_t hidden_size;
  float epsilon;
  DeviceView output;
};

struct MultiAxisRope final {
  DeviceView query;
  DeviceView key;
  DeviceView position_ids;
  int32_t tokens;
  int32_t query_heads;
  int32_t key_value_heads;
  int32_t head_dim;
  int32_t temporal_section;
  int32_t height_section;
  int32_t width_section;
  float frequency_base;
  DeviceView query_output;
  DeviceView key_output;
};

struct StoreKvCache final {
  DeviceView key;
  DeviceView value;
  int32_t token_count;
  int32_t key_value_size;
  int32_t start_position;
  int32_t cache_capacity;
  DeviceView key_cache;
  DeviceView value_cache;
  DeviceView cache_position;
};

struct Attention final {
  DeviceView query;
  DeviceView key_cache;
  DeviceView value_cache;
  int32_t query_token_count;
  int32_t query_head_count;
  int32_t key_value_head_count;
  int32_t cached_token_count;
  int32_t start_position;
  int32_t head_dim;
  bool is_causal;
  DeviceView output;
  DeviceView decode_partials;
  DeviceView cache_position;
};

struct AdvanceSequence final {
  DeviceView positions;
  DeviceView cache_position;
};

struct ResidualRmsNorm final {
  DeviceView input;
  DeviceView residual;
  DeviceView weight;
  int32_t rows;
  int32_t hidden_size;
  float epsilon;
  DeviceView residual_output;
  DeviceView output;
};

struct NormalizeRopeCache final {
  DeviceView query;
  DeviceView key;
  DeviceView value;
  DeviceView query_weight;
  DeviceView key_weight;
  DeviceView positions;
  DeviceView cache_position;
  int32_t query_heads;
  int32_t key_value_heads;
  int32_t head_dim;
  int32_t cache_capacity;
  int32_t temporal_section;
  int32_t height_section;
  int32_t width_section;
  float frequency_base;
  float epsilon;
  DeviceView query_output;
  DeviceView key_cache;
  DeviceView value_cache;
};

struct Add final {
  DeviceView lhs;
  DeviceView rhs;
  uint64_t element_count;
  DeviceView output;
};

struct SwiGLU final {
  DeviceView gate;
  DeviceView up;
  uint64_t element_count;
  DeviceView output;
};

struct Argmax final {
  DeviceView logits;
  int32_t rows;
  int32_t columns;
  DeviceView output;
};

struct ImagePatchPreprocess final {
  uint64_t pixel_count;
  DeviceView source;
  int32_t source_width_pixels;
  int32_t source_height_pixels;
  int32_t patch_size;
  DeviceView destination;
  int32_t input_width_pixels = 0;
  int32_t input_height_pixels = 0;
};

struct PairedPatchEmbedding final {
  DeviceView input;
  DeviceView first_weight;
  DeviceView second_weight;
  DeviceView bias;
  int32_t height;
  int32_t width;
  int32_t channels;
  int32_t output_channels;
  int32_t spatial_patch;
  DeviceView output;
};

struct GridPositionEmbedding final {
  DeviceView input;
  DeviceView position_weight;
  int32_t grid_height;
  int32_t grid_width;
  int32_t position_grid_size;
  int32_t hidden_size;
  int32_t merge_size;
  DeviceView output;
};

struct Bias final {
  DeviceView input;
  DeviceView bias;
  uint64_t element_count;
  int32_t row_size;
  DeviceView output;
};

struct BiasResidual final {
  DeviceView input;
  DeviceView bias;
  DeviceView residual;
  uint64_t element_count;
  int32_t row_size;
  DeviceView output;
};

struct GeluBias final {
  DeviceView input;
  DeviceView bias;
  uint64_t element_count;
  int32_t row_size;
  bool is_approximate = true;
  DeviceView output;
};

struct GridQkvRope final {
  DeviceView qkv;
  int32_t token_count;
  int32_t grid_height;
  int32_t grid_width;
  int32_t merge_size;
  int32_t head_count;
  int32_t head_size;
  DeviceView query;
  DeviceView key;
  DeviceView value;
};

struct PatchEmbedding final {
  DeviceView input;
  DeviceView weight;
  DeviceView bias;
  int32_t frames;
  int32_t height;
  int32_t width;
  int32_t channels;
  int32_t output_channels;
  int32_t temporal_patch;
  int32_t spatial_patch;
  DeviceView output;
};

struct PatchMerge final {
  DeviceView input;
  int32_t token_count;
  int32_t grid_width;
  int32_t hidden_size;
  int32_t merge_size;
  DeviceView output;
};

using ComputeOperation =
    std::variant<Embedding, RmsNorm, LayerNorm, MultiAxisRope, StoreKvCache, Attention, Add, SwiGLU,
                 Argmax, ImagePatchPreprocess, PairedPatchEmbedding, GridPositionEmbedding, Bias,
                 BiasResidual, GeluBias, GridQkvRope, Linear, AdvanceSequence, ResidualRmsNorm,
                 NormalizeRopeCache, AdaRmsNorm, FlowStep, GatedAdd, SplitGatedQuery, FusedFfn,
                 MultiHeadAttention, PatchEmbedding, PatchMerge, ParallelLinear>;

struct ComputeNode final {
  ComputeOperation operation;
  std::vector<uint32_t> dependencies;
};

} // namespace omni_runtime::backend
