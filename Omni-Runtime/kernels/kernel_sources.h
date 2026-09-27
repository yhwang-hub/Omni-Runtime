#pragma once

#include <string>

namespace omni_runtime {
namespace kernels {
namespace test {

inline std::string MakeAddFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/add_kernel/add_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace add_kernel {
__global__ void AddFloatKernel(const float *const __restrict__ lhs,
                               const float *const __restrict__ rhs, const uint64_t element_count,
                               float *const __restrict__ output) {
  AddImpl(lhs, rhs, element_count, output);
}
} // namespace add_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeLayernormFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/layernorm_kernel/layernorm_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace layernorm_kernel {
__global__ void LayernormFloatKernel(const float *const __restrict__ input,
                                     const float *const __restrict__ weight,
                                     const float *const __restrict__ bias, const int32_t rows,
                                     const int32_t hidden_size, const float epsilon,
                                     float *const __restrict__ output) {
  LayernormImpl(input, weight, bias, rows, hidden_size, epsilon, output);
}
} // namespace layernorm_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeFlashAttentionFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/flash_attention_kernel/flash_attention_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace flash_attention_kernel {
__global__ void FlashAttentionFloatKernel(const float *const query, const float *const key,
                                          const float *const value, const int32_t heads,
                                          const int32_t tokens, const int32_t head_dim,
                                          const bool is_causal, float *const output) {
  attention_impl::RunOnlineAttention(query, key, value, heads, tokens, head_dim, is_causal, output);
}
} // namespace flash_attention_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeGqaAttentionFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/gqa_attention_kernel/gqa_attention_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace gqa_attention_kernel {
__global__ void GqaAttentionFloatKernel(
    const float *const __restrict__ query, const float *const __restrict__ key_cache,
    const float *const __restrict__ value_cache, const int32_t query_token_count,
    const int32_t query_head_count, const int32_t key_value_head_count,
    const int32_t cached_token_count, const int32_t start_position, const int32_t head_dim,
    const bool is_causal, float *const __restrict__ output) {
  GqaAttentionImpl(query, key_cache, value_cache, query_token_count, query_head_count,
                   key_value_head_count, cached_token_count, start_position, head_dim, is_causal,
                   output);
}
} // namespace gqa_attention_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeStoreKvCacheFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/kv_cache_kernel/kv_cache_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace kv_cache_kernel {
__global__ void StoreKvCacheFloatKernel(const float *const __restrict__ key,
                                        const float *const __restrict__ value,
                                        const int32_t token_count, const int32_t key_value_size,
                                        const int32_t start_position, const int32_t cache_capacity,
                                        float *const __restrict__ key_cache,
                                        float *const __restrict__ value_cache) {
  StoreKvCacheImpl(key, value, token_count, key_value_size, start_position, cache_capacity,
                   key_cache, value_cache);
}
} // namespace kv_cache_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeFlowStepFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/floaw_step_kernel/flow_step_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace flow_step_kernel {
__global__ void FlowStepFloatKernel(const float *const __restrict__ state,
                                    const float *const __restrict__ velocity, const float dt,
                                    const uint64_t element_count,
                                    float *const __restrict__ output) {
  FlowStepImpl(state, velocity, dt, element_count, output);
}
} // namespace flow_step_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeQwen3VLPreprocessSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/image_kernel/preprocess.cuh>
extern "C" {
namespace omni_runtime {
namespace inference {
namespace example {
__global__ void Qwen3VLPreprocessKernel(const uint64_t pixel_count, const uint8_t *const source,
                                        const int32_t source_width_pixels,
                                        const int32_t source_height_pixels,
                                        const int32_t patch_size, uint16_t *const destination) {
  Qwen3VLPreprocessImpl(pixel_count, source, source_width_pixels, source_height_pixels, patch_size, destination);
}
} // namespace example
} // namespace inference
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeArgmaxFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/argmax_kernel/argmax_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace argmax_kernel {
__global__ void ArgmaxFloatKernel(const float *const __restrict__ logits, const int32_t rows,
                                  const int32_t columns, int32_t *const __restrict__ output) {
  ArgmaxImpl(logits, rows, columns, output);
}
} // namespace argmax_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeSwigluFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/swiglu_kernel/swiglu_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace swiglu_kernel {
__global__ void SwigluFloatKernel(const float *const __restrict__ gate,
                                  const float *const __restrict__ up, const uint64_t element_count,
                                  float *const __restrict__ output) {
  SwigluImpl(gate, up, element_count, output);
}
} // namespace swiglu_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeMropeFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/mrope_kernel/mrope_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace mrope_kernel {
__global__ void
MropeFloatKernel(const float *const __restrict__ query, const float *const __restrict__ key,
                 const int32_t *const __restrict__ position_ids, const int32_t tokens,
                 const int32_t query_heads, const int32_t key_value_heads, const int32_t head_dim,
                 const int32_t temporal_section, const int32_t height_section,
                 const int32_t width_section, const float frequency_base,
                 float *const __restrict__ query_output, float *const __restrict__ key_output) {
  MropeImpl(query, key, position_ids, tokens, query_heads, key_value_heads, head_dim,
            temporal_section, height_section, width_section, frequency_base, query_output,
            key_output);
}
} // namespace mrope_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeSplitGatedQueryFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/gated_attention_kernel/gated_attention_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace gated_attention_kernel {
__global__ void SplitGatedQueryFloatKernel(const float *const __restrict__ query_gate,
                                           const int32_t token_count, const int32_t head_count,
                                           const int32_t head_size, float *const __restrict__ query,
                                           float *const __restrict__ gate) {
  SplitGatedQueryImpl(query_gate, token_count, head_count, head_size, query, gate);
}
} // namespace gated_attention_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeMatmulFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/matmul_kernel/matmul_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace matmul_kernel {
__global__ void MatmulFloatKernel(const float *const __restrict__ lhs,
                                  const float *const __restrict__ rhs, const int32_t rows,
                                  const int32_t inner, const int32_t columns,
                                  float *const __restrict__ output) {
  MatmulFloatImpl(lhs, rhs, rows, inner, columns, output);
}
} // namespace matmul_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeEmbeddingFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/embedding_kernel/embedding_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace embedding_kernel {
__global__ void EmbeddingFloatKernel(const int32_t *const __restrict__ indices,
                                     const float *const __restrict__ weight,
                                     const int32_t token_count, const int32_t hidden_size,
                                     float *const __restrict__ output) {
  EmbeddingImpl(indices, weight, token_count, hidden_size, output);
}
} // namespace embedding_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeFusedFfnFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/fused_ffn_kernel/fused_ffn_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace fused_ffn_kernel {
__global__ void FusedFfnFloatKernel(const float *const __restrict__ input,
                                    const float *const __restrict__ fc1_weight,
                                    const float *const __restrict__ fc1_bias,
                                    const float *const __restrict__ fc2_weight,
                                    const float *const __restrict__ fc2_bias, const int32_t rows,
                                    const int32_t input_size, const int32_t intermediate_size,
                                    const int32_t output_size, float *const __restrict__ output) {
  FusedFfnImpl(input, fc1_weight, fc1_bias, fc2_weight, fc2_bias, rows, input_size,
               intermediate_size, output_size, output);
}
} // namespace fused_ffn_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeMhaFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/mha_kernel/mha_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace mha_kernel {
__global__ void MhaFloatKernel(const float *const query, const float *const key,
                               const float *const value, const int32_t heads, const int32_t tokens,
                               const int32_t head_dim, const bool is_causal, float *const output) {
  MhaFloatImpl(query, key, value, heads, tokens, head_dim, is_causal, output);
}
} // namespace mha_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeGatedAddFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/gated_add_kernel/gated_add_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace gated_add_kernel {
__global__ void GatedAddFloatKernel(const float *const __restrict__ lhs,
                                    const float *const __restrict__ rhs,
                                    const float *const __restrict__ gate,
                                    const uint64_t element_count, const int32_t hidden_size,
                                    float *const __restrict__ output) {
  GatedAddImpl(lhs, rhs, gate, element_count, hidden_size, output);
}
} // namespace gated_add_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeVisionPatchEmbedFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/vision_encoder_kernel/vision_encoder_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace vision_encoder_kernel {
__global__ void VisionPatchEmbedFloatKernel(
    const float *const __restrict__ input, const float *const __restrict__ weight,
    const float *const __restrict__ bias, const int32_t frames, const int32_t height,
    const int32_t width, const int32_t channels, const int32_t output_channels,
    const int32_t temporal_patch, const int32_t spatial_patch, float *const __restrict__ output) {
  VisionPatchEmbedImpl(input, weight, bias, frames, height, width, channels, output_channels,
                       temporal_patch, spatial_patch, output);
}
} // namespace vision_encoder_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

inline std::string MakeAdaRmsNormFloatSource() {
  return R"KERNEL_SOURCE(
#include <cstdint>
#include <kernels/adarms_kernel/adarms_kernel.cuh>
extern "C" {
namespace omni_runtime {
namespace kernels {
namespace adarms_kernel {
__global__ void AdaRmsNormFloatKernel(const float *const __restrict__ input,
                                      const float *const __restrict__ scale,
                                      const float *const __restrict__ shift, const int32_t rows,
                                      const int32_t hidden_size, const float epsilon,
                                      float *const __restrict__ output) {
  AdaRmsNormImpl(input, scale, shift, rows, hidden_size, epsilon, output);
}
} // namespace adarms_kernel
} // namespace kernels
} // namespace omni_runtime
}
)KERNEL_SOURCE";
}

} // namespace test
} // namespace kernels
} // namespace omni_runtime
