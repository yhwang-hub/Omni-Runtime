#include "Omni-Runtime/inference/tokenizer/byte_pair_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <utility>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace {

std::string EncodeCodePoint(const uint32_t code_point) {
  std::string output;
  if (code_point <= 0x7FU) {
    output.push_back(static_cast<char>(code_point));
  } else if (code_point <= 0x7FFU) {
    output.push_back(static_cast<char>(0xC0U | (code_point >> 6U)));
    output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
  } else if (code_point <= 0xFFFFU) {
    output.push_back(static_cast<char>(0xE0U | (code_point >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
  } else {
    output.push_back(static_cast<char>(0xF0U | (code_point >> 18U)));
    output.push_back(static_cast<char>(0x80U | ((code_point >> 12U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3FU)));
    output.push_back(static_cast<char>(0x80U | (code_point & 0x3FU)));
  }
  return output;
}

size_t Utf8CharacterSize(const unsigned char first_byte) {
  if ((first_byte & 0x80U) == 0U) {
    return 1U;
  }
  if ((first_byte & 0xE0U) == 0xC0U) {
    return 2U;
  }
  if ((first_byte & 0xF0U) == 0xE0U) {
    return 3U;
  }
  if ((first_byte & 0xF8U) == 0xF0U) {
    return 4U;
  }
  return 1U;
}

std::string MergeKey(const std::string &left, const std::string &right) {
  return left + '\0' + right;
}

bool IsAsciiLetter(const unsigned char character) {
  return std::isalpha(character) != 0;
}

bool IsAsciiDigit(const unsigned char character) {
  return std::isdigit(character) != 0;
}

bool IsNewline(const unsigned char character) {
  return character == '\r' || character == '\n';
}

bool IsWhitespace(const unsigned char character) {
  return std::isspace(character) != 0;
}

bool IsLetterLike(const unsigned char character) {
  return IsAsciiLetter(character) || character >= 0x80U;
}

} // namespace

bool BytePairTokenizer::Init(const TokenizerModel &model) {
  error_message_.clear();
  OMNI_RETURN_VAL_IF(model.type != TokenizerType::BYTE_PAIR,
                SetError("tokenizer model is not byte-pair encoding"));
  OMNI_RETURN_VAL_IF(model.vocabulary.empty(), SetError("byte-pair vocabulary is empty"));
  vocabulary_ = model.vocabulary;
  special_tokens_ = model.special_tokens;
  unknown_token_id_ = model.unknown_token_id;
  token_to_id_.clear();
  merge_ranks_.clear();
  byte_decoder_.clear();
  token_to_id_.reserve(vocabulary_.size());
  for (size_t token_id = 0U; token_id < vocabulary_.size(); ++token_id) {
    OMNI_RETURN_VAL_IF(token_id > static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                  SetError("byte-pair token id exceeds int32 range"));
    token_to_id_.emplace(vocabulary_[token_id], static_cast<int32_t>(token_id));
  }
  use_sentencepiece_marker_ = token_to_id_.find("\xE2\x96\x81") != token_to_id_.end();
  for (size_t rank = 0U; rank < model.merges.size(); ++rank) {
    const std::string &merge = model.merges[rank];
    const size_t separator = merge.find(' ');
    OMNI_RETURN_VAL_IF(separator == std::string::npos || separator == 0U ||
                      separator + 1U >= merge.size() ||
                      rank > static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                  SetError("invalid byte-pair merge rule"));
    merge_ranks_.emplace(MergeKey(merge.substr(0U, separator), merge.substr(separator + 1U)),
                         static_cast<int32_t>(rank));
  }

  std::vector<uint32_t> code_points;
  code_points.reserve(256U);
  for (uint32_t value = 33U; value <= 126U; ++value) {
    code_points.emplace_back(value);
  }
  for (uint32_t value = 161U; value <= 172U; ++value) {
    code_points.emplace_back(value);
  }
  for (uint32_t value = 174U; value <= 255U; ++value) {
    code_points.emplace_back(value);
  }
  uint32_t extra_code_point = 256U;
  for (uint32_t byte = 0U; byte <= 255U; ++byte) {
    const bool is_direct =
        std::find(code_points.begin(), code_points.end(), byte) != code_points.end();
    const uint32_t code_point = is_direct ? byte : extra_code_point++;
    byte_encoder_[byte] = EncodeCodePoint(code_point);
    byte_decoder_.emplace(byte_encoder_[byte], static_cast<uint8_t>(byte));
  }
  return true;
}

bool BytePairTokenizer::Encode(const std::string &text,
                               std::vector<int32_t> *const token_ids) const {
  OMNI_RETURN_VAL_IF(token_ids == nullptr, SetError("token id output is null"));
  token_ids->clear();
  size_t offset = 0U;
  size_t ordinary_begin = 0U;
  while (offset < text.size()) {
    const std::string *matched_special_text = nullptr;
    int32_t matched_special_id = -1;
    for (const auto &special_token : special_tokens_) {
      if (text.compare(offset, special_token.first.size(), special_token.first) == 0 &&
          (matched_special_text == nullptr ||
           special_token.first.size() > matched_special_text->size())) {
        matched_special_text = &special_token.first;
        matched_special_id = special_token.second;
      }
    }
    if (matched_special_text == nullptr) {
      ++offset;
      continue;
    }
    OMNI_RETURN_VAL_IF(!EncodeOrdinary(text.substr(ordinary_begin, offset - ordinary_begin), token_ids),
                  false);
    token_ids->emplace_back(matched_special_id);
    offset += matched_special_text->size();
    ordinary_begin = offset;
  }
  return EncodeOrdinary(text.substr(ordinary_begin), token_ids);
}

bool BytePairTokenizer::EncodeOrdinary(const std::string &text,
                                       std::vector<int32_t> *const token_ids) const {
  OMNI_RETURN_VAL_IF(token_ids == nullptr, SetError("token id output is null"));
  for (const std::string &piece : PreTokenize(text)) {
    OMNI_RETURN_VAL_IF(!EncodePiece(piece, token_ids), false);
  }
  return true;
}

bool BytePairTokenizer::EncodePiece(const std::string &piece,
                                    std::vector<int32_t> *const token_ids) const {
  OMNI_RETURN_VAL_IF(token_ids == nullptr, SetError("token id output is null"));
  if (piece.empty()) {
    return true;
  }
  std::string encoded_piece;
  if (use_sentencepiece_marker_) {
    constexpr char kSentencepieceMarker[] = "\xE2\x96\x81";
    for (const unsigned char byte : piece) {
      if (byte == static_cast<unsigned char>(' ')) {
        encoded_piece += kSentencepieceMarker;
      } else {
        encoded_piece.push_back(static_cast<char>(byte));
      }
    }
  } else {
    for (const unsigned char byte : piece) {
      encoded_piece += byte_encoder_[byte];
    }
  }
  const auto direct_token = token_to_id_.find(encoded_piece);
  if (direct_token != token_to_id_.end()) {
    token_ids->emplace_back(direct_token->second);
    return true;
  }

  std::vector<std::string> symbols;
  for (size_t offset = 0U; offset < encoded_piece.size();) {
    const size_t character_size =
        Utf8CharacterSize(static_cast<unsigned char>(encoded_piece[offset]));
    OMNI_RETURN_VAL_IF(offset + character_size > encoded_piece.size(),
                  SetError("invalid UTF-8 in byte-pair encoded text"));
    symbols.emplace_back(encoded_piece.substr(offset, character_size));
    offset += character_size;
  }
  while (symbols.size() > 1U) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best_index = symbols.size();
    for (size_t index = 0U; index + 1U < symbols.size(); ++index) {
      const auto rank = merge_ranks_.find(MergeKey(symbols[index], symbols[index + 1U]));
      if (rank != merge_ranks_.end() && rank->second < best_rank) {
        best_rank = rank->second;
        best_index = index;
      }
    }
    if (best_index == symbols.size()) {
      break;
    }
    symbols[best_index] += symbols[best_index + 1U];
    symbols.erase(symbols.begin() + static_cast<ptrdiff_t>(best_index + 1U));
  }
  for (const std::string &symbol : symbols) {
    const auto token = token_to_id_.find(symbol);
    if (token != token_to_id_.end()) {
      token_ids->emplace_back(token->second);
    } else if (unknown_token_id_ >= 0) {
      token_ids->emplace_back(unknown_token_id_);
    } else {
      return SetError("byte-pair symbol is not in vocabulary");
    }
  }
  return true;
}

std::vector<std::string> BytePairTokenizer::PreTokenize(const std::string &text) const {
  std::vector<std::string> pieces;
  if (use_sentencepiece_marker_) {
    size_t offset = 0U;
    while (offset < text.size()) {
      if (text[offset] != ' ') {
        const size_t begin = offset;
        while (offset < text.size() && text[offset] != ' ') {
          ++offset;
        }
        pieces.emplace_back(text.substr(begin, offset - begin));
        continue;
      }
      const size_t begin = offset;
      while (offset < text.size() && text[offset] == ' ') {
        ++offset;
      }
      if (offset == text.size()) {
        pieces.emplace_back(text.substr(begin, offset - begin));
        break;
      }
      while (offset < text.size() && text[offset] != ' ') {
        ++offset;
      }
      pieces.emplace_back(text.substr(begin, offset - begin));
    }
    return pieces;
  }
  size_t offset = 0U;
  while (offset < text.size()) {
    const size_t begin = offset;
    const unsigned char current = static_cast<unsigned char>(text[offset]);
    if (current == '\'' && offset + 1U < text.size()) {
      const std::string suffix = text.substr(offset, std::min<size_t>(4U, text.size() - offset));
      static const std::vector<std::string> contractions = {"'s",  "'t",  "'m", "'d",
                                                            "'re", "'ve", "'ll"};
      size_t contraction_size = 0U;
      for (const std::string &contraction : contractions) {
        if (suffix.size() >= contraction.size() &&
            std::equal(contraction.begin(), contraction.end(), suffix.begin(),
                       [](const char left, const char right) {
                         return std::tolower(static_cast<unsigned char>(left)) ==
                                std::tolower(static_cast<unsigned char>(right));
                       })) {
          contraction_size = std::max(contraction_size, contraction.size());
        }
      }
      if (contraction_size > 0U) {
        offset += contraction_size;
        pieces.emplace_back(text.substr(begin, offset - begin));
        continue;
      }
    }
    if (!IsLetterLike(current) && !IsAsciiDigit(current) && !IsNewline(current) &&
        offset + 1U < text.size() && IsLetterLike(static_cast<unsigned char>(text[offset + 1U]))) {
      ++offset;
      while (offset < text.size() && IsLetterLike(static_cast<unsigned char>(text[offset]))) {
        offset += Utf8CharacterSize(static_cast<unsigned char>(text[offset]));
      }
    } else if (IsLetterLike(current)) {
      while (offset < text.size() && IsLetterLike(static_cast<unsigned char>(text[offset]))) {
        offset += Utf8CharacterSize(static_cast<unsigned char>(text[offset]));
      }
    } else if (IsAsciiDigit(current)) {
      ++offset;
    } else if (IsNewline(current)) {
      while (offset < text.size() && IsNewline(static_cast<unsigned char>(text[offset]))) {
        ++offset;
      }
    } else if (IsWhitespace(current)) {
      while (offset < text.size() && IsWhitespace(static_cast<unsigned char>(text[offset])) &&
             !IsNewline(static_cast<unsigned char>(text[offset]))) {
        ++offset;
      }
    } else {
      ++offset;
      while (offset < text.size()) {
        const unsigned char character = static_cast<unsigned char>(text[offset]);
        if (IsWhitespace(character) || IsLetterLike(character) || IsAsciiDigit(character)) {
          break;
        }
        ++offset;
      }
      while (offset < text.size() && IsNewline(static_cast<unsigned char>(text[offset]))) {
        ++offset;
      }
    }
    pieces.emplace_back(text.substr(begin, offset - begin));
  }
  return pieces;
}

bool BytePairTokenizer::Decode(const std::vector<int32_t> &token_ids,
                               std::string *const text) const {
  OMNI_RETURN_VAL_IF(text == nullptr, SetError("decoded text output is null"));
  text->clear();
  for (const int32_t token_id : token_ids) {
    OMNI_RETURN_VAL_IF(token_id < 0 || static_cast<size_t>(token_id) >= vocabulary_.size(),
                  SetError("token id is out of vocabulary range"));
    const std::string &token = vocabulary_[static_cast<size_t>(token_id)];
    if (special_tokens_.find(token) != special_tokens_.end()) {
      *text += token;
      continue;
    }
    if (use_sentencepiece_marker_) {
      constexpr char kSentencepieceMarker[] = "\xE2\x96\x81";
      for (size_t offset = 0U; offset < token.size();) {
        if (offset + sizeof(kSentencepieceMarker) - 1U <= token.size() &&
            token.compare(offset, sizeof(kSentencepieceMarker) - 1U, kSentencepieceMarker) == 0) {
          text->push_back(' ');
          offset += sizeof(kSentencepieceMarker) - 1U;
        } else {
          text->push_back(token[offset]);
          ++offset;
        }
      }
      continue;
    }
    for (size_t offset = 0U; offset < token.size();) {
      const size_t character_size = Utf8CharacterSize(static_cast<unsigned char>(token[offset]));
      OMNI_RETURN_VAL_IF(offset + character_size > token.size(),
                    SetError("invalid UTF-8 in tokenizer vocabulary"));
      const std::string character = token.substr(offset, character_size);
      const auto byte = byte_decoder_.find(character);
      OMNI_RETURN_VAL_IF(byte == byte_decoder_.end(), SetError("tokenizer byte decoder is incomplete"));
      text->push_back(static_cast<char>(byte->second));
      offset += character_size;
    }
  }
  return true;
}

bool BytePairTokenizer::SetError(const std::string &message) const {
  error_message_ = message;
  return false;
}

const std::string &BytePairTokenizer::error_message() const {
  return error_message_;
}

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
