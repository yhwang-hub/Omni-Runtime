#include "Omni-Runtime/inference/tokenizer/unigram_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {

bool UnigramTokenizer::Init(const TokenizerModel &model) {
  error_message_.clear();
  OMNI_RETURN_VAL_IF(model.type != TokenizerType::UNIGRAM, SetError("tokenizer model is not Unigram"));
  OMNI_RETURN_VAL_IF(model.vocabulary.empty(), SetError("Unigram vocabulary is empty"));
  vocabulary_ = model.vocabulary;
  scores_ = model.scores;
  if (scores_.size() != vocabulary_.size()) {
    scores_.assign(vocabulary_.size(), 0.0f);
  }
  unknown_token_id_ = model.unknown_token_id;
  token_to_id_.clear();
  maximum_token_size_bytes_ = 0U;
  for (size_t token_id = 0U; token_id < vocabulary_.size(); ++token_id) {
    OMNI_RETURN_VAL_IF(token_id > static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                  SetError("Unigram token id exceeds int32 range"));
    token_to_id_.emplace(vocabulary_[token_id], static_cast<int32_t>(token_id));
    maximum_token_size_bytes_ = std::max(maximum_token_size_bytes_, vocabulary_[token_id].size());
  }
  return true;
}

bool UnigramTokenizer::Encode(const std::string &text,
                              std::vector<int32_t> *const token_ids) const {
  OMNI_RETURN_VAL_IF(token_ids == nullptr, SetError("token id output is null"));
  token_ids->clear();
  const size_t text_size = text.size();
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  std::vector<float> best_scores(text_size + 1U, negative_infinity);
  std::vector<int32_t> best_tokens(text_size + 1U, -1);
  std::vector<size_t> previous_offsets(text_size + 1U, 0U);
  best_scores[0U] = 0.0f;
  for (size_t begin = 0U; begin < text_size; ++begin) {
    if (!std::isfinite(best_scores[begin])) {
      continue;
    }
    const size_t maximum_end = std::min(text_size, begin + maximum_token_size_bytes_);
    for (size_t end = begin + 1U; end <= maximum_end; ++end) {
      const auto token = token_to_id_.find(text.substr(begin, end - begin));
      if (token == token_to_id_.end()) {
        continue;
      }
      const float score = best_scores[begin] + scores_[static_cast<size_t>(token->second)];
      if (score > best_scores[end]) {
        best_scores[end] = score;
        best_tokens[end] = token->second;
        previous_offsets[end] = begin;
      }
    }
    if (best_tokens[begin + 1U] < 0 && unknown_token_id_ >= 0) {
      best_scores[begin + 1U] = best_scores[begin] - 100.0f;
      best_tokens[begin + 1U] = unknown_token_id_;
      previous_offsets[begin + 1U] = begin;
    }
  }
  OMNI_RETURN_VAL_IF(best_tokens[text_size] < 0, SetError("Unigram input cannot be segmented"));
  std::vector<int32_t> reversed_tokens;
  for (size_t offset = text_size; offset > 0U; offset = previous_offsets[offset]) {
    OMNI_RETURN_VAL_IF(best_tokens[offset] < 0 || previous_offsets[offset] >= offset,
                  SetError("Unigram segmentation state is invalid"));
    reversed_tokens.emplace_back(best_tokens[offset]);
  }
  token_ids->assign(reversed_tokens.rbegin(), reversed_tokens.rend());
  return true;
}

bool UnigramTokenizer::Decode(const std::vector<int32_t> &token_ids,
                              std::string *const text) const {
  OMNI_RETURN_VAL_IF(text == nullptr, SetError("decoded text output is null"));
  text->clear();
  for (const int32_t token_id : token_ids) {
    OMNI_RETURN_VAL_IF(token_id < 0 || static_cast<size_t>(token_id) >= vocabulary_.size(),
                  SetError("Unigram token id is out of range"));
    *text += vocabulary_[static_cast<size_t>(token_id)];
  }
  return true;
}

bool UnigramTokenizer::SetError(const std::string &message) const {
  error_message_ = message;
  return false;
}

const std::string &UnigramTokenizer::error_message() const {
  return error_message_;
}

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
