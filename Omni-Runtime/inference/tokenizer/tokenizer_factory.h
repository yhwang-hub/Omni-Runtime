#pragma once

#include <memory>

#include "Omni-Runtime/inference/tokenizer/tokenizer.h"

namespace omni_runtime {
namespace inference {

class TokenizerFactory final {
public:
  static std::unique_ptr<Tokenizer> Create(const TokenizerType type);
};

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
