#include "Omni-Runtime/models/qwen3_5/weight_schema.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_5 {
bool Qwen35WeightSchema::Build(const util::GGUFSerializer &model,
                               const util::GGUFSerializer &vision,
                               const TokenizerModel &tokenizer) {
  error_message_.clear();
  model_config_ = Qwen35Config{};
  weights_.clear();
  OMNI_RETURN_VAL_IF(!LoadMetadata(model, vision), false);
  OMNI_RETURN_VAL_IF(tokenizer.vocabulary.empty() ||
                         tokenizer.vocabulary.size() >
                             static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                     SetError("tokenizer vocabulary is empty or exceeds int32 range"));
  model_config_.vocabulary_size = static_cast<int32_t>(tokenizer.vocabulary.size());
  const std::pair<std::string, int32_t *> special[] = {
      {"<|im_end|>", &model_config_.end_of_sequence_id},
      {"<|endoftext|>", &model_config_.end_of_text_id},
      {"<|image_pad|>", &model_config_.image_id},
  };
  for (const auto &[name, id] : special) {
    const auto token = tokenizer.special_tokens.find(name);
    OMNI_RETURN_VAL_IF(token == tokenizer.special_tokens.end(),
                       SetError("missing tokenizer special token: " + name));
    *id = token->second;
  }
  OMNI_RETURN_VAL_IF(!RegisterWeights(model) || !RegisterWeights(vision), false);
  OMNI_RETURN_VAL_IF(!ValidateWeights(), SetError("incompatible GGUF tensor registry"));
  return true;
}

bool Qwen35WeightSchema::LoadMetadata(const util::GGUFSerializer &model,
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
    OMNI_RETURN_VAL_IF(!model.GetIntegerMetadata("qwen35." + key, output) || *output <= 0,
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
  OMNI_RETURN_VAL_IF(!qwen3::LoadVisionMetadata(vision, &v) || v.image_size <= 0 ||
                         v.patch_size <= 0 || v.hidden_size <= 0 || v.feed_forward_size <= 0 ||
                         v.layer_count <= 0 || v.head_count <= 0 || v.projection_size <= 0 ||
                         v.merge_size <= 0,
                     SetError("missing or invalid vision metadata"));
  OMNI_RETURN_VAL_IF(v.hidden_size % v.head_count != 0 || v.projection_size != c.hidden_size ||
                         v.head_size() > 256 || v.head_size() % 4 != 0 ||
                         v.image_size % v.patch_size != 0 ||
                         static_cast<int64_t>(v.hidden_size) * v.merge_size * v.merge_size >
                             std::numeric_limits<int32_t>::max() ||
                         v.hidden_size > std::numeric_limits<int32_t>::max() / 3 ||
                         !std::isfinite(v.layer_norm_epsilon) || v.layer_norm_epsilon <= 0.0F,
                     SetError("invalid vision dimensions or normalization epsilon"));
  return true;
}

bool Qwen35WeightSchema::RegisterWeights(const util::GGUFSerializer &serializer) {
  for (const auto &tensor : serializer.model().tensors) {
    if (tensor.name.ends_with(".input_scale") || tensor.name.ends_with(".scale")) {
      continue;
    }
    const auto type = static_cast<util::GGUFTensorType>(tensor.type_id);
    OMNI_RETURN_VAL_IF(
        type != util::GGUFTensorType::FLOAT32 && type != util::GGUFTensorType::FLOAT16 &&
            type != util::GGUFTensorType::Q8_0 && type != util::GGUFTensorType::BFLOAT16 &&
            type != util::GGUFTensorType::NVFP4 && type != util::GGUFTensorType::FP8_E4M3,
        SetError("unsupported GGUF tensor type for " + tensor.name));
    OMNI_RETURN_VAL_IF(weights_.contains(tensor.name),
                       SetError("duplicate model/vision tensor name: " + tensor.name));
    WeightStorage weight;
    weight.type = type;
    weight.dimensions = tensor.dimensions;
    if (type == util::GGUFTensorType::FP8_E4M3) {
      OMNI_RETURN_VAL_IF(!tensor.name.ends_with(".weight") || tensor.dimensions.size() != 2U,
                         SetError("invalid FP8 weight: " + tensor.name));
      const std::string scale_name =
          tensor.name.substr(0U, tensor.name.size() - 7U) + ".weight_scale_inv";
      const auto *const scale_tensor = serializer.FindTensor(scale_name);
      const std::vector<uint64_t> scale_shape = {tensor.dimensions[0] / 128U,
                                                 tensor.dimensions[1] / 128U};
      OMNI_RETURN_VAL_IF(tensor.dimensions[0] % 128U != 0U || tensor.dimensions[1] % 128U != 0U ||
                             scale_tensor == nullptr ||
                             scale_tensor->type_id !=
                                 static_cast<uint32_t>(util::GGUFTensorType::FLOAT32) ||
                             scale_tensor->dimensions != scale_shape,
                         SetError("missing or incompatible FP8 block scales: " + tensor.name));
      std::vector<float> scales;
      OMNI_RETURN_VAL_IF(
          !serializer.ReadTensorData<float>(
              scale_name, 0U, static_cast<size_t>(scale_tensor->element_count), &scales) ||
              !std::all_of(scales.begin(), scales.end(),
                           [](const float value) { return std::isfinite(value) && value > 0.0F; }),
          SetError("invalid FP8 block scales: " + scale_name));
      model_config_.has_fp8_weights = true;
    }
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
    OMNI_RETURN_VAL_IF(!tensor.has_known_storage_size,
                       SetError("unknown GGUF storage size: " + tensor.name));
    weights_.emplace(tensor.name, std::move(weight));
  }
  return true;
}

bool Qwen35WeightSchema::ValidateWeights() const {
  const auto &c = model_config_;
  const auto check = [this](const std::string &name, const std::vector<uint64_t> &shape,
                            const bool is_float) {
    const auto found = weights_.find(name);
    OMNI_RETURN_VAL_IF_LOG(found == weights_.end() || found->second.dimensions != shape ||
                               (is_float && found->second.type != util::GGUFTensorType::FLOAT32),
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
              !check(prefix + (c.has_fp8_weights ? "ssm_a_log" : "ssm_a"),
                     {static_cast<uint64_t>(c.recurrent_value_heads)}, true) ||
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

} // namespace qwen3_5
} // namespace inference
} // namespace omni_runtime
