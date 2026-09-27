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

// GGUF storage types are independent of the activation scalar type.
struct QuantizedLinear final {
  DeviceView input;
  DeviceView weight;
  int32_t weight_type;
  float weight_scale;
  int32_t rows;
  int32_t inner;
  int32_t columns;
  DeviceView output;
  DeviceView activation_scales;
  int32_t activation_block_size = 32;
  bool is_packed_nvfp4 = false;
  bool is_ggml_split_k = false;
  bool is_packed_q8 = false;
};

struct QuantizeActivation final {
  // NVFP4 prefill stores row-unscaled values or, with is_packed_nvfp4, GGUF
  // 36-byte blocks of 64 values. Both forms repeat the row scale per 16 values.
  DeviceView input;
  int32_t weight_type;
  int32_t rows;
  int32_t columns;
  bool is_decode;
  DeviceView output;
  DeviceView scales;
  bool is_ggml_rounding = false;
  bool is_packed_nvfp4 = false;
  // Store signed byte codes rather than codes expanded to the activation scalar type.
  bool is_packed_q8 = false;
};

struct QuantizedEmbedding final {
  DeviceView indices;
  DeviceView weight;
  int32_t weight_type;
  int32_t token_count;
  int32_t hidden_size;
  int32_t vocabulary_size;
  DeviceView output;
};

struct CausalConv1d final {
  DeviceView input;
  DeviceView weight;
  DeviceView state;
  int32_t token_count;
  int32_t channels;
  int32_t kernel_size;
  DeviceView output;
};

struct GatedDeltaNet final {
  DeviceView qkv;
  DeviceView alpha;
  DeviceView beta;
  DeviceView decay_weight;
  DeviceView time_bias;
  DeviceView state;
  int32_t token_count;
  int32_t key_head_count;
  int32_t value_head_count;
  int32_t head_size;
  float epsilon;
  DeviceView output;
};

struct SigmoidGate final {
  DeviceView input;
  DeviceView gate;
  uint64_t element_count;
  DeviceView output;
};

struct PartialMultiAxisRope final {
  DeviceView query;
  DeviceView key;
  DeviceView position_ids;
  int32_t tokens;
  int32_t query_heads;
  int32_t key_value_heads;
  int32_t head_dim;
  int32_t rotary_dim;
  int32_t temporal_section;
  int32_t height_section;
  int32_t width_section;
  float frequency_base;
  DeviceView query_output;
  DeviceView key_output;
};

struct RmsNorm final {
  DeviceView input;
  DeviceView weight;
  int32_t rows;
  int32_t hidden_size;
  float epsilon;
  DeviceView output;
  bool is_ggml_normalization = false;
};

struct LayerNorm final {
  DeviceView input;
  DeviceView weight;
  DeviceView bias;
  int32_t rows;
  int32_t hidden_size;
  float epsilon;
  DeviceView output;
  bool is_ggml_normalization = false;
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
  bool is_fp16_cache = false;
  bool is_fp16_key_value = false;
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
  bool is_fp16_attention = false;
  bool is_fp16_accumulation = false;
  bool is_fp16_key_value = false;
  bool is_ggml_softmax = false;
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
  bool is_ggml_normalization = false;
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
  bool is_ggml_silu = false;
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
  bool is_letterbox = false;
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
  bool is_fp16_projection = false;
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
  bool is_ggml_interpolation = false;
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
  bool is_ggml_gelu = false;
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
  bool is_fp16_key_value = false;
  bool is_ggml_rope = false;
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
                 MultiHeadAttention, PatchEmbedding, PatchMerge, ParallelLinear, QuantizedLinear,
                 QuantizedEmbedding, CausalConv1d, GatedDeltaNet, SigmoidGate, PartialMultiAxisRope,
                 QuantizeActivation>;

struct ComputeNode final {
  ComputeOperation operation;
  std::vector<uint32_t> dependencies;
};

} // namespace omni_runtime::backend
