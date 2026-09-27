#include "Omni-Runtime/models/qwen3_vl/weight_schema.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <utility>

#include "Omni-Runtime/models/components/qwen_vision/config.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {
bool Qwen3VLWeightSchema::Build(const util::GGUFSerializer &model_serializer,
                                const util::GGUFSerializer &vision_serializer) {
  error_message_.clear();
  model_weights_.clear();
  vision_weights_.clear();
  output_weight_name_.clear();
  Qwen3VLConfig model_config;
  qwen3::Qwen3VisionConfig vision_config;
  std::string architecture;
  OMNI_RETURN_VAL_IF(!model_serializer.GetMetadata("general.architecture", &architecture) ||
                         architecture != "qwen3vl",
                     SetError("GGUF architecture is not qwen3vl"));
  OMNI_RETURN_VAL_IF(
      !model_serializer.GetIntegerMetadata("qwen3vl.block_count", &model_config.layer_count) ||
          !model_serializer.GetIntegerMetadata("qwen3vl.embedding_length",
                                               &model_config.hidden_size) ||
          !model_serializer.GetIntegerMetadata("qwen3vl.feed_forward_length",
                                               &model_config.feed_forward_size) ||
          !model_serializer.GetIntegerMetadata("qwen3vl.attention.head_count",
                                               &model_config.query_head_count) ||
          !model_serializer.GetIntegerMetadata("qwen3vl.attention.head_count_kv",
                                               &model_config.key_value_head_count) ||
          !model_serializer.GetIntegerMetadata("qwen3vl.attention.key_length",
                                               &model_config.head_size),
      SetError("Qwen3-VL integer metadata is incomplete"));
  const util::GGUFMetadata *const tokens_metadata =
      model_serializer.FindMetadata("tokenizer.ggml.tokens");
  OMNI_RETURN_VAL_IF(tokens_metadata == nullptr ||
                         tokens_metadata->value.type != util::GGUFValueType::ARRAY ||
                         tokens_metadata->value.array.size() >
                             static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                     SetError("Qwen3-VL tokenizer vocabulary is invalid"));
  model_config.vocabulary_size = static_cast<int32_t>(tokens_metadata->value.array.size());
  OMNI_RETURN_VAL_IF(
      !model_serializer.GetIntegerMetadata("qwen3vl.n_deepstack_layers",
                                           &model_config.deepstack_layer_count) ||
          !model_serializer.GetFloatMetadata("qwen3vl.rope.freq_base",
                                             &model_config.rope_frequency_base) ||
          !model_serializer.GetFloatMetadata("qwen3vl.attention.layer_norm_rms_epsilon",
                                             &model_config.rms_norm_epsilon) ||
          !model_serializer.GetMetadataArray("qwen3vl.rope.dimension_sections",
                                             &model_config.mrope_sections),
      SetError("Qwen3-VL model metadata is incomplete"));
  model_serializer.GetMetadata("tokenizer.ggml.eos_token_id",
                               &model_config.special_tokens.end_of_sequence_id);
  model_config_ = model_config;
  OMNI_RETURN_VAL_IF(!BuildModelRegistry(model_serializer, model_config), false);

  OMNI_RETURN_VAL_IF(!qwen3::LoadVisionMetadata(vision_serializer, &vision_config),
                     SetError("Qwen3-VL vision metadata is incomplete"));
  for (const util::GGUFTensorInfo &tensor : vision_serializer.model().tensors) {
    constexpr char kPrefix[] = "v.deepstack.";
    constexpr char kSuffix[] = ".norm.weight";
    if (tensor.name.starts_with(kPrefix) && tensor.name.ends_with(kSuffix) &&
        tensor.name.size() > sizeof(kPrefix) + sizeof(kSuffix) - 2U) {
      const std::string layer_text =
          tensor.name.substr(sizeof(kPrefix) - 1U,
                             tensor.name.size() - (sizeof(kPrefix) - 1U) - (sizeof(kSuffix) - 1U));
      int32_t layer = 0;
      const auto parsed =
          std::from_chars(layer_text.data(), layer_text.data() + layer_text.size(), layer);
      OMNI_RETURN_VAL_IF(parsed.ec != std::errc{} ||
                             parsed.ptr != layer_text.data() + layer_text.size() || layer < 0,
                         SetError("invalid Qwen3-VL DeepStack layer: " + tensor.name));
      vision_config.deepstack_layers.emplace_back(layer);
    }
  }
  std::sort(vision_config.deepstack_layers.begin(), vision_config.deepstack_layers.end());
  vision_config.deepstack_layers.erase(
      std::unique(vision_config.deepstack_layers.begin(), vision_config.deepstack_layers.end()),
      vision_config.deepstack_layers.end());
  vision_config_ = vision_config;
  model_config_.merge_size = vision_config.merge_size;
  OMNI_RETURN_VAL_IF(!BuildVisionRegistry(vision_serializer), false);
  return true;
}

bool Qwen3VLWeightSchema::BuildModelRegistry(const util::GGUFSerializer &serializer,
                                             const Qwen3VLConfig &config) {
  const uint64_t hidden = config.hidden_size;
  const uint64_t head = config.head_size;
  const uint64_t query = static_cast<uint64_t>(config.query_head_count) * head;
  const uint64_t kv = static_cast<uint64_t>(config.key_value_head_count) * head;
  const uint64_t ffn = config.feed_forward_size;
  model_weights_ = {{"token_embd.weight", {hidden, static_cast<uint64_t>(config.vocabulary_size)}},
                    {"output_norm.weight", {hidden}}};
  output_weight_name_ =
      serializer.HasTensor("output.weight") ? "output.weight" : "token_embd.weight";
  if (output_weight_name_ == "output.weight") {
    model_weights_.push_back(
        {output_weight_name_, {hidden, static_cast<uint64_t>(config.vocabulary_size)}});
  }
  for (const auto &weight : model_weights_) {
    OMNI_RETURN_VAL_IF(!ValidateTensor(serializer, weight), false);
  }
  const Qwen3VLWeightName layer_weights[] = {
      {"attn_norm.weight", {hidden}},          {"attn_q_norm.weight", {head}},
      {"attn_k_norm.weight", {head}},          {"attn_q.weight", {hidden, query}},
      {"attn_k.weight", {hidden, kv}},         {"attn_v.weight", {hidden, kv}},
      {"attn_output.weight", {query, hidden}}, {"ffn_norm.weight", {hidden}},
      {"ffn_gate.weight", {hidden, ffn}},      {"ffn_up.weight", {hidden, ffn}},
      {"ffn_down.weight", {ffn, hidden}},
  };
  for (int32_t layer = 0; layer < config.layer_count; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    for (const auto &entry : layer_weights) {
      Qwen3VLWeightName weight{prefix + entry.name, entry.dimensions};
      OMNI_RETURN_VAL_IF(!ValidateTensor(serializer, weight), false);
      model_weights_.emplace_back(std::move(weight));
    }
  }
  return true;
}

bool Qwen3VLWeightSchema::BuildVisionRegistry(const util::GGUFSerializer &serializer) {
  for (const util::GGUFTensorInfo &tensor : serializer.model().tensors) {
    if (tensor.name == "v.patch_embd.weight" || tensor.name == "v.patch_embd.weight.1") {
      continue;
    }
    Qwen3VLWeightName weight{tensor.name, tensor.dimensions};
    OMNI_RETURN_VAL_IF(!ValidateTensor(serializer, weight), false);
    vision_weights_.emplace_back(std::move(weight));
  }
  for (const std::string name : {"v.patch_embd.weight", "v.patch_embd.weight.1"}) {
    Qwen3VLWeightName weight{name,
                             {16U, 16U, 3U, static_cast<uint64_t>(vision_config_.hidden_size)}};
    OMNI_RETURN_VAL_IF(!ValidateTensor(serializer, weight), false);
    vision_weights_.emplace_back(std::move(weight));
  }
  return true;
}

bool Qwen3VLWeightSchema::ValidateTensor(const util::GGUFSerializer &serializer,
                                         const Qwen3VLWeightName &weight) {
  const util::GGUFTensorInfo *const tensor = serializer.FindTensor(weight.name);
  OMNI_RETURN_VAL_IF(tensor == nullptr, SetError("missing GGUF tensor: " + weight.name));
  OMNI_RETURN_VAL_IF(tensor->dimensions != weight.dimensions,
                     SetError("invalid dimensions for GGUF tensor: " + weight.name));
  return true;
}

bool Qwen3VLWeightSchema::SetError(const std::string &message) {
  error_message_ = message;
  return false;
}

const Qwen3VLConfig &Qwen3VLWeightSchema::model_config() const {
  return model_config_;
}
const qwen3::Qwen3VisionConfig &Qwen3VLWeightSchema::vision_config() const {
  return vision_config_;
}
const std::string &Qwen3VLWeightSchema::output_weight_name() const {
  return output_weight_name_;
}
const std::vector<Qwen3VLWeightName> &Qwen3VLWeightSchema::model_weights() const {
  return model_weights_;
}
const std::vector<Qwen3VLWeightName> &Qwen3VLWeightSchema::vision_weights() const {
  return vision_weights_;
}
const std::string &Qwen3VLWeightSchema::error_message() const {
  return error_message_;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
