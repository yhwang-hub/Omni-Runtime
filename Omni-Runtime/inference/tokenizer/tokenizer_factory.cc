#include "Omni-Runtime/inference/tokenizer/tokenizer_factory.h"

#include "Omni-Runtime/inference/tokenizer/byte_pair_tokenizer.h"
#include "Omni-Runtime/inference/tokenizer/unigram_tokenizer.h"
#include "Omni-Runtime/inference/tokenizer/word_piece_tokenizer.h"

namespace omni_runtime {
namespace inference {

std::unique_ptr<Tokenizer> TokenizerFactory::Create(const TokenizerType type) {
  switch (type) {
  case TokenizerType::BYTE_PAIR:
    return std::make_unique<BytePairTokenizer>();
  case TokenizerType::WORD_PIECE:
    return std::make_unique<WordPieceTokenizer>();
  case TokenizerType::UNIGRAM:
    return std::make_unique<UnigramTokenizer>();
  default:
    return nullptr;
  }
}

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
