#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Omni-Runtime/inference/tokenizer/tokenizer_define.h"

namespace omni_runtime {
namespace inference {

class Tokenizer {
public:
  virtual ~Tokenizer();

  virtual bool Init(const TokenizerModel &model) = 0;
  virtual bool Encode(const std::string &text, std::vector<int32_t> *const token_ids) const = 0;
  virtual bool Decode(const std::vector<int32_t> &token_ids, std::string *const text) const = 0;
  virtual const std::string &error_message() const = 0;
};

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
