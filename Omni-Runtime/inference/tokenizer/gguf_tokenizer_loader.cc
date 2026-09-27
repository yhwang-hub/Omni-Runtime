#include "Omni-Runtime/inference/tokenizer/gguf_tokenizer_loader.h"

#include <cstdint>
#include <limits>
#include <vector>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace {

// GGUF token type ids that must never be split by the byte-pair merges.
constexpr int32_t kControlTokenType = 3;
constexpr int32_t kUserDefinedTokenType = 4;

bool IsSpecialToken(const std::string &token) {
  return token.size() >= 4U && token[0] == '<' && token[1] == '|' &&
         token[token.size() - 2U] == '|' && token.back() == '>';
}

bool IsSpecialTokenType(const std::vector<int32_t> &token_types, const size_t token_id) {
  OMNI_RETURN_VAL_IF(token_id >= token_types.size(), false);
  const int32_t token_type = token_types[token_id];
  return token_type == kControlTokenType || token_type == kUserDefinedTokenType;
}

} // namespace

bool GGUFTokenizerLoader::Load(const util::GGUFSerializer &serializer,
                               TokenizerModel *const model) {
  error_message_.clear();
  OMNI_RETURN_VAL_IF(model == nullptr, SetError("tokenizer model output is null"));
  OMNI_RETURN_VAL_IF(!serializer.IsParsed(), SetError("GGUF model is not parsed"));
  TokenizerModel loaded_model;
  std::string tokenizer_type;
  OMNI_RETURN_VAL_IF(!serializer.GetMetadata("tokenizer.ggml.model", &tokenizer_type),
                SetError("GGUF tokenizer model type is missing"));
  if (tokenizer_type == "gpt2" || tokenizer_type == "bpe") {
    loaded_model.type = TokenizerType::BYTE_PAIR;
  } else if (tokenizer_type == "wordpiece") {
    loaded_model.type = TokenizerType::WORD_PIECE;
  } else if (tokenizer_type == "llama" || tokenizer_type == "unigram") {
    loaded_model.type = TokenizerType::UNIGRAM;
  } else {
    return SetError("unsupported GGUF tokenizer model: " + tokenizer_type);
  }
  OMNI_RETURN_VAL_IF(!serializer.GetMetadataArray("tokenizer.ggml.tokens", &loaded_model.vocabulary) ||
                    loaded_model.vocabulary.empty(),
                SetError("GGUF tokenizer vocabulary is missing"));
  if (serializer.HasMetadata("tokenizer.ggml.merges")) {
    serializer.GetMetadataArray("tokenizer.ggml.merges", &loaded_model.merges);
  }
  if (serializer.HasMetadata("tokenizer.ggml.scores")) {
    serializer.GetMetadataArray("tokenizer.ggml.scores", &loaded_model.scores);
  }

  std::vector<int32_t> token_types;
  if (serializer.HasMetadata("tokenizer.ggml.token_type")) {
    serializer.GetMetadataArray("tokenizer.ggml.token_type", &token_types);
  }
  for (size_t token_id = 0U; token_id < loaded_model.vocabulary.size(); ++token_id) {
    const std::string &token = loaded_model.vocabulary[token_id];
    if (IsSpecialToken(token) || IsSpecialTokenType(token_types, token_id)) {
      OMNI_RETURN_VAL_IF(token_id > static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                    SetError("GGUF special token id exceeds int32 range"));
      loaded_model.special_tokens.emplace(token, static_cast<int32_t>(token_id));
    }
  }
  uint32_t unknown_token_id = 0U;
  if (serializer.HasMetadata("tokenizer.ggml.unknown_token_id") &&
      serializer.GetMetadata("tokenizer.ggml.unknown_token_id", &unknown_token_id) &&
      unknown_token_id <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    loaded_model.unknown_token_id = static_cast<int32_t>(unknown_token_id);
  }
  uint32_t bos_token_id = 0U;
  if (serializer.HasMetadata("tokenizer.ggml.bos_token_id") &&
      serializer.GetMetadata("tokenizer.ggml.bos_token_id", &bos_token_id) &&
      bos_token_id <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    loaded_model.bos_token_id = static_cast<int32_t>(bos_token_id);
  }
  if (serializer.HasMetadata("tokenizer.ggml.add_bos_token")) {
    serializer.GetMetadata("tokenizer.ggml.add_bos_token", &loaded_model.add_bos_token);
  }
  *model = std::move(loaded_model);
  return true;
}

bool GGUFTokenizerLoader::SetError(const std::string &message) {
  error_message_ = message;
  return false;
}

const std::string &GGUFTokenizerLoader::error_message() const {
  return error_message_;
}

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
