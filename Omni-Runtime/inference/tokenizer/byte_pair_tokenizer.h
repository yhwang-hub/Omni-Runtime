#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/inference/tokenizer/tokenizer.h"

namespace omni_runtime {
namespace inference {

class BytePairTokenizer final : public Tokenizer {
public:
  bool Init(const TokenizerModel &model) final;
  bool Encode(const std::string &text, std::vector<int32_t> *const token_ids) const final;
  bool Decode(const std::vector<int32_t> &token_ids, std::string *const text) const final;
  const std::string &error_message() const final;

private:
  bool EncodeOrdinary(const std::string &text, std::vector<int32_t> *const token_ids) const;
  bool EncodePiece(const std::string &piece, std::vector<int32_t> *const token_ids) const;
  bool SetError(const std::string &message) const;
  std::vector<std::string> PreTokenize(const std::string &text) const;

private:
  std::vector<std::string> vocabulary_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  std::unordered_map<std::string, int32_t> merge_ranks_;
  std::unordered_map<std::string, int32_t> special_tokens_;
  std::array<std::string, 256U> byte_encoder_ = {};
  std::unordered_map<std::string, uint8_t> byte_decoder_;
  int32_t unknown_token_id_ = -1;
  bool use_sentencepiece_marker_ = false;
  mutable std::string error_message_;
};

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
