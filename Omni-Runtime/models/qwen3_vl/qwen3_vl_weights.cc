#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_weights.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {
namespace {

template <typename T>
bool GetIntegerMetadata(const util::GGUFSerializer &serializer, const std::string &key,
                        T *const output) {
  OMNI_RETURN_VAL_IF(output == nullptr, false);
  uint32_t uint32_value = 0U;
  if (serializer.GetMetadata(key, &uint32_value) &&
      uint32_value <= static_cast<uint32_t>(std::numeric_limits<T>::max())) {
    *output = static_cast<T>(uint32_value);
    return true;
  }
  int32_t int32_value = 0;
  return serializer.GetMetadata(key, &int32_value) && int32_value >= 0 &&
         static_cast<uint32_t>(int32_value) <=
             static_cast<uint32_t>(std::numeric_limits<T>::max()) &&
         (*output = static_cast<T>(int32_value), true);
}

bool GetFloatMetadata(const util::GGUFSerializer &serializer, const std::string &key,
                      float *const output) {
  OMNI_RETURN_VAL_IF(output == nullptr, false);
  if (serializer.GetMetadata(key, output)) {
    return true;
  }
  double value = 0.0;
  OMNI_RETURN_VAL_IF(!serializer.GetMetadata(key, &value), false);
  *output = static_cast<float>(value);
  return true;
}

Qwen3VLWeightName MakeWeight(const std::string &name, const std::vector<uint64_t> &dimensions) {
  return Qwen3VLWeightName{name, dimensions};
}

} // namespace

bool Qwen3VLWeightRegistry::Build(const util::GGUFSerializer &model_serializer,
                                  const util::GGUFSerializer &vision_serializer) {
  error_message_.clear();
  layers_.clear();
  vision_weights_.clear();
  has_output_weight_ = false;
  Qwen3VLConfig model_config;
  Qwen3VLVisionConfig vision_config;
  std::string architecture;
  OMNI_RETURN_VAL_IF(!model_serializer.GetMetadata("general.architecture", &architecture) ||
                         architecture != "qwen3vl",
                     SetError("GGUF architecture is not qwen3vl"));
  OMNI_RETURN_VAL_IF(
      !GetIntegerMetadata(model_serializer, "qwen3vl.block_count", &model_config.layer_count) ||
          !GetIntegerMetadata(model_serializer, "qwen3vl.embedding_length",
                              &model_config.hidden_size) ||
          !GetIntegerMetadata(model_serializer, "qwen3vl.feed_forward_length",
                              &model_config.feed_forward_size) ||
          !GetIntegerMetadata(model_serializer, "qwen3vl.attention.head_count",
                              &model_config.query_head_count) ||
          !GetIntegerMetadata(model_serializer, "qwen3vl.attention.head_count_kv",
                              &model_config.key_value_head_count) ||
          !GetIntegerMetadata(model_serializer, "qwen3vl.attention.key_length",
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
  OMNI_RETURN_VAL_IF(!GetIntegerMetadata(model_serializer, "qwen3vl.n_deepstack_layers",
                                         &model_config.deepstack_layer_count) ||
                         !GetFloatMetadata(model_serializer, "qwen3vl.rope.freq_base",
                                           &model_config.rope_frequency_base) ||
                         !GetFloatMetadata(model_serializer,
                                           "qwen3vl.attention.layer_norm_rms_epsilon",
                                           &model_config.rms_norm_epsilon) ||
                         !model_serializer.GetMetadataArray("qwen3vl.rope.dimension_sections",
                                                            &model_config.mrope_sections),
                     SetError("Qwen3-VL model metadata is incomplete"));
  model_serializer.GetMetadata("tokenizer.ggml.eos_token_id",
                               &model_config.special_tokens.end_of_sequence_id);
  model_config_ = model_config;
  OMNI_RETURN_VAL_IF(!BuildModelRegistry(model_serializer, model_config), false);

  OMNI_RETURN_VAL_IF(
      !GetIntegerMetadata(vision_serializer, "clip.vision.image_size", &vision_config.image_size) ||
          !GetIntegerMetadata(vision_serializer, "clip.vision.patch_size",
                              &vision_config.patch_size) ||
          !GetIntegerMetadata(vision_serializer, "clip.vision.embedding_length",
                              &vision_config.hidden_size) ||
          !GetIntegerMetadata(vision_serializer, "clip.vision.feed_forward_length",
                              &vision_config.feed_forward_size) ||
          !GetIntegerMetadata(vision_serializer, "clip.vision.block_count",
                              &vision_config.layer_count) ||
          !GetIntegerMetadata(vision_serializer, "clip.vision.attention.head_count",
                              &vision_config.head_count) ||
          !GetIntegerMetadata(vision_serializer, "clip.vision.projection_dim",
                              &vision_config.projection_size) ||
          !GetIntegerMetadata(vision_serializer, "clip.vision.spatial_merge_size",
                              &vision_config.merge_size) ||
          !vision_serializer.GetMetadata("clip.vision.attention.layer_norm_epsilon",
                                         &vision_config.layer_norm_epsilon),
      SetError("Qwen3-VL vision metadata is incomplete"));
  for (const util::GGUFTensorInfo &tensor : vision_serializer.model().tensors) {
    constexpr char kPrefix[] = "v.deepstack.";
    constexpr char kSuffix[] = ".norm.weight";
    if (tensor.name.starts_with(kPrefix) && tensor.name.ends_with(kSuffix) &&
        tensor.name.size() > sizeof(kPrefix) + sizeof(kSuffix) - 2U) {
      const std::string layer_text =
          tensor.name.substr(sizeof(kPrefix) - 1U,
                             tensor.name.size() - (sizeof(kPrefix) - 1U) - (sizeof(kSuffix) - 1U));
      vision_config.deepstack_layers.emplace_back(std::stoi(layer_text));
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

bool Qwen3VLWeightRegistry::BuildModelRegistry(const util::GGUFSerializer &serializer,
                                               const Qwen3VLConfig &config) {
  const auto dimension = [](const int32_t value) {
    return std::vector<uint64_t>{static_cast<uint64_t>(value)};
  };
  OMNI_RETURN_VAL_IF(!ValidateTensor(serializer, "token_embd.weight",
                                     {static_cast<uint64_t>(config.hidden_size),
                                      static_cast<uint64_t>(config.vocabulary_size)},
                                     &token_embedding_) ||
                         !ValidateTensor(serializer, "output_norm.weight",
                                         dimension(config.hidden_size), &output_norm_),
                     false);
  has_output_weight_ = serializer.HasTensor("output.weight");
  if (has_output_weight_) {
    OMNI_RETURN_VAL_IF(!ValidateTensor(serializer, "output.weight",
                                       {static_cast<uint64_t>(config.hidden_size),
                                        static_cast<uint64_t>(config.vocabulary_size)},
                                       &output_weight_),
                       false);
  } else {
    output_weight_ = token_embedding_;
  }

  layers_.reserve(static_cast<size_t>(config.layer_count));
  const int32_t query_size = config.query_head_count * config.head_size;
  const int32_t key_value_size = config.key_value_head_count * config.head_size;
  for (int32_t layer = 0; layer < config.layer_count; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    Qwen3VLLayerWeightNames names;
    const bool is_valid = ValidateTensor(serializer, prefix + "attn_norm.weight",
                                         dimension(config.hidden_size), &names.attention_norm) &&
                          ValidateTensor(serializer, prefix + "attn_q_norm.weight",
                                         dimension(config.head_size), &names.query_norm) &&
                          ValidateTensor(serializer, prefix + "attn_k_norm.weight",
                                         dimension(config.head_size), &names.key_norm) &&
                          ValidateTensor(serializer, prefix + "attn_q.weight",
                                         {static_cast<uint64_t>(config.hidden_size),
                                          static_cast<uint64_t>(query_size)},
                                         &names.query) &&
                          ValidateTensor(serializer, prefix + "attn_k.weight",
                                         {static_cast<uint64_t>(config.hidden_size),
                                          static_cast<uint64_t>(key_value_size)},
                                         &names.key) &&
                          ValidateTensor(serializer, prefix + "attn_v.weight",
                                         {static_cast<uint64_t>(config.hidden_size),
                                          static_cast<uint64_t>(key_value_size)},
                                         &names.value) &&
                          ValidateTensor(serializer, prefix + "attn_output.weight",
                                         {static_cast<uint64_t>(query_size),
                                          static_cast<uint64_t>(config.hidden_size)},
                                         &names.attention_output) &&
                          ValidateTensor(serializer, prefix + "ffn_norm.weight",
                                         dimension(config.hidden_size), &names.feed_forward_norm) &&
                          ValidateTensor(serializer, prefix + "ffn_gate.weight",
                                         {static_cast<uint64_t>(config.hidden_size),
                                          static_cast<uint64_t>(config.feed_forward_size)},
                                         &names.feed_forward_gate) &&
                          ValidateTensor(serializer, prefix + "ffn_up.weight",
                                         {static_cast<uint64_t>(config.hidden_size),
                                          static_cast<uint64_t>(config.feed_forward_size)},
                                         &names.feed_forward_up) &&
                          ValidateTensor(serializer, prefix + "ffn_down.weight",
                                         {static_cast<uint64_t>(config.feed_forward_size),
                                          static_cast<uint64_t>(config.hidden_size)},
                                         &names.feed_forward_down);
    OMNI_RETURN_VAL_IF(!is_valid, false);
    layers_.emplace_back(std::move(names));
  }
  return true;
}

bool Qwen3VLWeightRegistry::BuildVisionRegistry(const util::GGUFSerializer &serializer) {
  for (const util::GGUFTensorInfo &tensor : serializer.model().tensors) {
    if (tensor.name == "v.patch_embd.weight" || tensor.name == "v.patch_embd.weight.1") {
      continue;
    }
    Qwen3VLWeightName weight;
    OMNI_RETURN_VAL_IF(!ValidateTensor(serializer, tensor.name, tensor.dimensions, &weight), false);
    vision_weights_.emplace_back(std::move(weight));
  }
  Qwen3VLWeightName first_patch_weight;
  Qwen3VLWeightName second_patch_weight;
  OMNI_RETURN_VAL_IF(
      !ValidateTensor(serializer, "v.patch_embd.weight",
                      {16U, 16U, 3U, static_cast<uint64_t>(vision_config_.hidden_size)},
                      &first_patch_weight) ||
          !ValidateTensor(serializer, "v.patch_embd.weight.1",
                          {16U, 16U, 3U, static_cast<uint64_t>(vision_config_.hidden_size)},
                          &second_patch_weight),
      false);
  vision_weights_.emplace_back(std::move(first_patch_weight));
  vision_weights_.emplace_back(std::move(second_patch_weight));
  return true;
}

bool Qwen3VLWeightRegistry::ValidateTensor(const util::GGUFSerializer &serializer,
                                           const std::string &name,
                                           const std::vector<uint64_t> &dimensions,
                                           Qwen3VLWeightName *const output) {
  OMNI_RETURN_VAL_IF(output == nullptr, false);
  const util::GGUFTensorInfo *const tensor = serializer.FindTensor(name);
  OMNI_RETURN_VAL_IF(tensor == nullptr, SetError("missing GGUF tensor: " + name));
  OMNI_RETURN_VAL_IF(tensor->dimensions != dimensions,
                     SetError("invalid dimensions for GGUF tensor: " + name));
  *output = MakeWeight(name, dimensions);
  return true;
}

bool Qwen3VLWeightRegistry::SetError(const std::string &message) {
  error_message_ = message;
  return false;
}

const Qwen3VLConfig &Qwen3VLWeightRegistry::model_config() const {
  return model_config_;
}
const Qwen3VLVisionConfig &Qwen3VLWeightRegistry::vision_config() const {
  return vision_config_;
}
const Qwen3VLWeightName &Qwen3VLWeightRegistry::token_embedding() const {
  return token_embedding_;
}
const Qwen3VLWeightName &Qwen3VLWeightRegistry::output_norm() const {
  return output_norm_;
}
const Qwen3VLWeightName &Qwen3VLWeightRegistry::output_weight() const {
  return output_weight_;
}
const std::vector<Qwen3VLLayerWeightNames> &Qwen3VLWeightRegistry::layers() const {
  return layers_;
}
const std::vector<Qwen3VLWeightName> &Qwen3VLWeightRegistry::vision_weights() const {
  return vision_weights_;
}
bool Qwen3VLWeightRegistry::has_output_weight() const {
  return has_output_weight_;
}
const std::string &Qwen3VLWeightRegistry::error_message() const {
  return error_message_;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
