#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace omni_runtime {
namespace inference {

enum class TokenizerType : uint8_t {
  BYTE_PAIR = 0,
  WORD_PIECE = 1,
  UNIGRAM = 2,
};

struct TokenizerModel final {
  TokenizerType type = TokenizerType::BYTE_PAIR;
  std::vector<std::string> vocabulary;
  std::vector<std::string> merges;
  std::vector<float> scores;
  std::unordered_map<std::string, int32_t> special_tokens;
  int32_t unknown_token_id = -1;
  int32_t bos_token_id = -1;
  bool add_bos_token = false;
};

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
