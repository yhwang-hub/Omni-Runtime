#include "Omni-Runtime/models/qwen3_8_nvfp4/qwen3_8_nvfp4_pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime::inference::qwen3_8_nvfp4 {
namespace {

bool Integer(const util::GGUFSerializer &serializer, const std::string &key, int32_t *const value) {
  OMNI_RETURN_VAL_IF(value == nullptr, false);
  uint32_t number = 0U;
  OMNI_RETURN_VAL_IF(!serializer.GetMetadata(key, &number) ||
                         number > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()),
                     false);
  *value = static_cast<int32_t>(number);
  return true;
}

} // namespace

bool Qwen38Pipeline::LoadMetadata(const util::GGUFSerializer &model,
                                  const util::GGUFSerializer &vision) {
  auto &c = model_config_;
  auto &v = c.vision;
  std::string architecture;
  std::string projector;
  OMNI_RETURN_VAL_IF(
      !model.GetMetadata("general.architecture", &architecture) || architecture != "qwen35" ||
          !vision.GetMetadata("clip.projector_type", &projector) || projector != "qwen3vl_merger",
      SetError("expected qwen35 language and qwen3vl_merger vision GGUFs"));
  const std::pair<std::string, int32_t *> integers[] = {
      {"block_count", &c.layer_count},
      {"embedding_length", &c.hidden_size},
      {"feed_forward_length", &c.feed_forward_size},
      {"attention.head_count", &c.query_head_count},
      {"attention.head_count_kv", &c.key_value_head_count},
      {"attention.key_length", &c.head_size},
      {"rope.dimension_count", &c.rotary_size},
      {"ssm.conv_kernel", &c.convolution_size},
      {"ssm.state_size", &c.recurrent_head_size},
      {"ssm.group_count", &c.recurrent_key_heads},
      {"ssm.time_step_rank", &c.recurrent_value_heads},
      {"full_attention_interval", &c.full_attention_interval},
  };
  for (const auto &[key, output] : integers) {
    OMNI_RETURN_VAL_IF(!Integer(model, "qwen35." + key, output) || *output <= 0,
                       SetError("missing or invalid GGUF metadata: qwen35." + key));
  }
  OMNI_RETURN_VAL_IF(
      !model.GetMetadata("qwen35.rope.freq_base", &c.rope_frequency_base) ||
          !model.GetMetadata("qwen35.attention.layer_norm_rms_epsilon", &c.rms_norm_epsilon) ||
          !model.GetMetadataArray("qwen35.rope.dimension_sections", &c.mrope_sections) ||
          c.mrope_sections.size() != 4U || c.mrope_sections[3] != 0 ||
          c.mrope_sections[0] + c.mrope_sections[1] + c.mrope_sections[2] != c.rotary_size / 2 ||
          c.rotary_size > c.head_size || c.rotary_size % 2 != 0 ||
          !std::isfinite(c.rope_frequency_base) || c.rope_frequency_base <= 0.0F ||
          !std::isfinite(c.rms_norm_epsilon) || c.rms_norm_epsilon <= 0.0F ||
          c.query_head_count % c.key_value_head_count != 0 ||
          c.recurrent_value_heads % c.recurrent_key_heads != 0 || c.head_size > 256 ||
          c.hidden_size % 64 != 0 || c.feed_forward_size % 64 != 0 || c.mrope_sections[0] < 0 ||
          c.mrope_sections[1] < 0 || c.mrope_sections[2] < 0 ||
          static_cast<int64_t>(c.query_head_count) * c.head_size >
              std::numeric_limits<int32_t>::max() / 2 ||
          (2LL * c.recurrent_key_heads + c.recurrent_value_heads) * c.recurrent_head_size >
              std::numeric_limits<int32_t>::max() ||
          c.recurrent_head_size > 128 || c.recurrent_head_size % 8 != 0 || c.convolution_size < 2 ||
          c.convolution_size > 16,
      SetError("unsupported qwen35 rotary or recurrent dimensions"));
  c.recurrent_layers.resize(c.layer_count);
  for (int32_t layer = 0; layer < c.layer_count; ++layer) {
    c.recurrent_layers[layer] = (layer + 1) % c.full_attention_interval != 0;
  }
  if (model.HasMetadata("qwen35.attention.recurrent_layers")) {
    OMNI_RETURN_VAL_IF(
        !model.GetMetadataArray("qwen35.attention.recurrent_layers", &c.recurrent_layers) ||
            c.recurrent_layers.size() != static_cast<size_t>(c.layer_count),
        SetError("invalid qwen35 recurrent layer map"));
  }
  const std::pair<std::string, int32_t *> vision_integers[] = {
      {"image_size", &v.image_size},          {"patch_size", &v.patch_size},
      {"embedding_length", &v.hidden_size},   {"feed_forward_length", &v.feed_forward_size},
      {"block_count", &v.layer_count},        {"attention.head_count", &v.head_count},
      {"projection_dim", &v.projection_size}, {"spatial_merge_size", &v.merge_size},
  };
  for (const auto &[key, output] : vision_integers) {
    OMNI_RETURN_VAL_IF(!Integer(vision, "clip.vision." + key, output) || *output <= 0,
                       SetError("missing vision metadata: clip.vision." + key));
  }
  OMNI_RETURN_VAL_IF(
      !vision.GetMetadata("clip.vision.attention.layer_norm_epsilon", &v.layer_norm_epsilon) ||
          v.hidden_size % v.head_count != 0 || v.projection_size != c.hidden_size ||
          v.head_size() > 256 || v.head_size() % 4 != 0 || v.image_size % v.patch_size != 0 ||
          static_cast<int64_t>(v.hidden_size) * v.merge_size * v.merge_size >
              std::numeric_limits<int32_t>::max() ||
          v.hidden_size > std::numeric_limits<int32_t>::max() / 3 ||
          !std::isfinite(v.layer_norm_epsilon) || v.layer_norm_epsilon <= 0.0F,
      SetError("invalid vision dimensions or normalization epsilon"));
  return true;
}

bool Qwen38Pipeline::LoadWeights(const util::GGUFSerializer &serializer) {
  for (const auto &tensor : serializer.model().tensors) {
    if (tensor.name.ends_with(".input_scale") || tensor.name.ends_with(".scale")) {
      continue;
    }
    OMNI_RETURN_VAL_IF(tensor.type_id != 0U && tensor.type_id != 1U && tensor.type_id != 8U &&
                           tensor.type_id != 30U && tensor.type_id != 40U,
                       SetError("unsupported GGUF tensor type for " + tensor.name));
    OMNI_RETURN_VAL_IF(weights_.contains(tensor.name),
                       SetError("duplicate model/vision tensor name: " + tensor.name));
    Weight weight;
    weight.type = static_cast<int32_t>(tensor.type_id);
    weight.dimensions = tensor.dimensions;
    if (tensor.name.ends_with(".weight")) {
      const std::string scale_name = tensor.name.substr(0U, tensor.name.size() - 7U) + ".scale";
      if (serializer.HasTensor(scale_name)) {
        std::vector<float> scale;
        OMNI_RETURN_VAL_IF(!serializer.ReadTensorData<float>(scale_name, 0U, 1U, &scale) ||
                               scale.size() != 1U || !std::isfinite(scale[0]) || scale[0] <= 0.0F,
                           SetError("invalid GGUF tensor scale: " + scale_name));
        weight.scale = scale[0];
      }
    }
    std::vector<uint8_t> bytes;
    OMNI_RETURN_VAL_IF(
        !tensor.has_known_storage_size ||
            !serializer.ReadTensorBytes(tensor.name, 0U, tensor.storage_size_bytes, &bytes),
        SetError("failed to read GGUF tensor: " + tensor.name));
    weight.storage = memory::Buffer::CreateDeviceBufferFromVector(backend_, bytes);
    OMNI_RETURN_VAL_IF(weight.storage == nullptr,
                       SetError("failed to upload GGUF tensor: " + tensor.name));
    gpu_memory_bytes_ += bytes.size();
    weights_.emplace(tensor.name, std::move(weight));
  }
  return true;
}

bool Qwen38Pipeline::ValidateWeights() const {
  const auto &c = model_config_;
  const auto check = [this](const std::string &name, const std::vector<uint64_t> &shape,
                            const bool is_float) {
    const auto found = weights_.find(name);
    OMNI_RETURN_VAL_IF_LOG(found == weights_.end() || found->second.dimensions != shape ||
                               (is_float && found->second.type != 0),
                           false, ERROR, "Missing or incompatible tensor: " << name);
    return true;
  };
  OMNI_RETURN_VAL_IF(
      !check("token_embd.weight",
             {static_cast<uint64_t>(c.hidden_size), static_cast<uint64_t>(c.vocabulary_size)},
             false) ||
          !check("output.weight",
                 {static_cast<uint64_t>(c.hidden_size), static_cast<uint64_t>(c.vocabulary_size)},
                 false) ||
          !check("output_norm.weight", {static_cast<uint64_t>(c.hidden_size)}, true),
      false);
  const uint64_t hidden = c.hidden_size;
  const uint64_t recurrent = static_cast<uint64_t>(c.recurrent_value_heads) * c.recurrent_head_size;
  const uint64_t qkv =
      recurrent + 2U * static_cast<uint64_t>(c.recurrent_key_heads) * c.recurrent_head_size;
  for (int32_t layer = 0; layer < c.layer_count; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    OMNI_RETURN_VAL_IF(!check(prefix + "attn_norm.weight", {hidden}, true) ||
                           !check(prefix + "post_attention_norm.weight", {hidden}, true) ||
                           !check(prefix + "ffn_gate.weight",
                                  {hidden, static_cast<uint64_t>(c.feed_forward_size)}, false) ||
                           !check(prefix + "ffn_up.weight",
                                  {hidden, static_cast<uint64_t>(c.feed_forward_size)}, false) ||
                           !check(prefix + "ffn_down.weight",
                                  {static_cast<uint64_t>(c.feed_forward_size), hidden}, false),
                       false);
    if (c.IsRecurrent(layer)) {
      OMNI_RETURN_VAL_IF(
          !check(prefix + "attn_qkv.weight", {hidden, qkv}, false) ||
              !check(prefix + "attn_gate.weight", {hidden, recurrent}, false) ||
              !check(prefix + "ssm_out.weight", {recurrent, hidden}, false) ||
              !check(prefix + "ssm_conv1d.weight", {static_cast<uint64_t>(c.convolution_size), qkv},
                     true) ||
              !check(prefix + "ssm_norm.weight", {static_cast<uint64_t>(c.recurrent_head_size)},
                     true) ||
              !check(prefix + "ssm_a", {static_cast<uint64_t>(c.recurrent_value_heads)}, true) ||
              !check(prefix + "ssm_dt.bias", {static_cast<uint64_t>(c.recurrent_value_heads)},
                     true) ||
              !check(prefix + "ssm_alpha.weight",
                     {hidden, static_cast<uint64_t>(c.recurrent_value_heads)}, false) ||
              !check(prefix + "ssm_beta.weight",
                     {hidden, static_cast<uint64_t>(c.recurrent_value_heads)}, false),
          false);
    } else {
      const uint64_t query = static_cast<uint64_t>(c.query_head_count) * c.head_size;
      const uint64_t kv = static_cast<uint64_t>(c.key_value_head_count) * c.head_size;
      OMNI_RETURN_VAL_IF(
          !check(prefix + "attn_q.weight", {hidden, query * 2U}, false) ||
              !check(prefix + "attn_k.weight", {hidden, kv}, false) ||
              !check(prefix + "attn_v.weight", {hidden, kv}, false) ||
              !check(prefix + "attn_output.weight", {query, hidden}, false) ||
              !check(prefix + "attn_q_norm.weight", {static_cast<uint64_t>(c.head_size)}, true) ||
              !check(prefix + "attn_k_norm.weight", {static_cast<uint64_t>(c.head_size)}, true),
          false);
    }
  }
  const auto &v = c.vision;
  const uint64_t vision_hidden = v.hidden_size;
  const uint64_t vision_ffn = v.feed_forward_size;
  const uint64_t merged_hidden = v.merged_hidden_size();
  const uint64_t position_grid = v.image_size / v.patch_size;
  const std::vector<uint64_t> patch_shape = {
      static_cast<uint64_t>(v.patch_size), static_cast<uint64_t>(v.patch_size), 3U, vision_hidden};
  OMNI_RETURN_VAL_IF(
      !check("v.patch_embd.weight", patch_shape, true) ||
          !check("v.patch_embd.weight.1", patch_shape, true) ||
          !check("v.patch_embd.bias", {vision_hidden}, true) ||
          !check("v.position_embd.weight", {vision_hidden, position_grid * position_grid}, true) ||
          !check("v.post_ln.weight", {vision_hidden}, true) ||
          !check("v.post_ln.bias", {vision_hidden}, true) ||
          !check("mm.0.weight", {merged_hidden, merged_hidden}, false) ||
          !check("mm.0.bias", {merged_hidden}, true) ||
          !check("mm.2.weight", {merged_hidden, static_cast<uint64_t>(c.hidden_size)}, false) ||
          !check("mm.2.bias", {static_cast<uint64_t>(c.hidden_size)}, true),
      false);
  for (int32_t layer = 0; layer < v.layer_count; ++layer) {
    const std::string prefix = "v.blk." + std::to_string(layer) + ".";
    OMNI_RETURN_VAL_IF(
        !check(prefix + "ln1.weight", {vision_hidden}, true) ||
            !check(prefix + "ln1.bias", {vision_hidden}, true) ||
            !check(prefix + "ln2.weight", {vision_hidden}, true) ||
            !check(prefix + "ln2.bias", {vision_hidden}, true) ||
            !check(prefix + "attn_qkv.weight", {vision_hidden, 3U * vision_hidden}, false) ||
            !check(prefix + "attn_qkv.bias", {3U * vision_hidden}, true) ||
            !check(prefix + "attn_out.weight", {vision_hidden, vision_hidden}, false) ||
            !check(prefix + "attn_out.bias", {vision_hidden}, true) ||
            !check(prefix + "ffn_up.weight", {vision_hidden, vision_ffn}, false) ||
            !check(prefix + "ffn_up.bias", {vision_ffn}, true) ||
            !check(prefix + "ffn_down.weight", {vision_ffn, vision_hidden}, false) ||
            !check(prefix + "ffn_down.bias", {vision_hidden}, true),
        false);
  }
  return true;
}

} // namespace omni_runtime::inference::qwen3_8_nvfp4
