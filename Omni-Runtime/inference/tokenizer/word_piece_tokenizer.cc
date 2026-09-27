#include "Omni-Runtime/inference/tokenizer/word_piece_tokenizer.h"

#include <cctype>
#include <limits>
#include <sstream>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {

bool WordPieceTokenizer::Init(const TokenizerModel &model) {
  error_message_.clear();
  OMNI_RETURN_VAL_IF(model.type != TokenizerType::WORD_PIECE,
                SetError("tokenizer model is not WordPiece"));
  OMNI_RETURN_VAL_IF(model.vocabulary.empty(), SetError("WordPiece vocabulary is empty"));
  vocabulary_ = model.vocabulary;
  unknown_token_id_ = model.unknown_token_id;
  token_to_id_.clear();
  for (size_t token_id = 0U; token_id < vocabulary_.size(); ++token_id) {
    OMNI_RETURN_VAL_IF(token_id > static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                  SetError("WordPiece token id exceeds int32 range"));
    token_to_id_.emplace(vocabulary_[token_id], static_cast<int32_t>(token_id));
  }
  return true;
}

bool WordPieceTokenizer::Encode(const std::string &text,
                                std::vector<int32_t> *const token_ids) const {
  OMNI_RETURN_VAL_IF(token_ids == nullptr, SetError("token id output is null"));
  token_ids->clear();
  std::istringstream stream(text);
  std::string word;
  while (stream >> word) {
    size_t begin = 0U;
    bool is_unknown = false;
    while (begin < word.size()) {
      size_t end = word.size();
      int32_t matched_token = -1;
      size_t matched_end = begin;
      while (end > begin) {
        const std::string prefix = begin == 0U ? "" : "##";
        const auto token = token_to_id_.find(prefix + word.substr(begin, end - begin));
        if (token != token_to_id_.end()) {
          matched_token = token->second;
          matched_end = end;
          break;
        }
        --end;
      }
      if (matched_token < 0) {
        is_unknown = true;
        break;
      }
      token_ids->emplace_back(matched_token);
      begin = matched_end;
    }
    if (is_unknown) {
      OMNI_RETURN_VAL_IF(unknown_token_id_ < 0, SetError("WordPiece word is not in vocabulary"));
      token_ids->emplace_back(unknown_token_id_);
    }
  }
  return true;
}

bool WordPieceTokenizer::Decode(const std::vector<int32_t> &token_ids,
                                std::string *const text) const {
  OMNI_RETURN_VAL_IF(text == nullptr, SetError("decoded text output is null"));
  text->clear();
  for (const int32_t token_id : token_ids) {
    OMNI_RETURN_VAL_IF(token_id < 0 || static_cast<size_t>(token_id) >= vocabulary_.size(),
                  SetError("WordPiece token id is out of range"));
    const std::string &token = vocabulary_[static_cast<size_t>(token_id)];
    if (token.compare(0U, 2U, "##") == 0) {
      *text += token.substr(2U);
    } else {
      if (!text->empty()) {
        text->push_back(' ');
      }
      *text += token;
    }
  }
  return true;
}

bool WordPieceTokenizer::SetError(const std::string &message) const {
  error_message_ = message;
  return false;
}

const std::string &WordPieceTokenizer::error_message() const {
  return error_message_;
}

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
