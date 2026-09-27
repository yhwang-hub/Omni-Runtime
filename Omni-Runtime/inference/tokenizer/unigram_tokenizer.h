#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/inference/tokenizer/tokenizer.h"

namespace omni_runtime {
namespace inference {

class UnigramTokenizer final : public Tokenizer {
public:
  bool Init(const TokenizerModel &model) final;
  bool Encode(const std::string &text, std::vector<int32_t> *const token_ids) const final;
  bool Decode(const std::vector<int32_t> &token_ids, std::string *const text) const final;
  const std::string &error_message() const final;

private:
  bool SetError(const std::string &message) const;

private:
  std::vector<std::string> vocabulary_;
  std::vector<float> scores_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  size_t maximum_token_size_bytes_ = 0U;
  int32_t unknown_token_id_ = -1;
  mutable std::string error_message_;
};

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
