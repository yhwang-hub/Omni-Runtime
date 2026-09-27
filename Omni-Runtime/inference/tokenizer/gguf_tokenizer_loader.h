#pragma once

#include <string>

#include "Omni-Runtime/inference/tokenizer/tokenizer_define.h"
#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"

namespace omni_runtime {
namespace inference {

class GGUFTokenizerLoader final {
public:
  bool Load(const util::GGUFSerializer &serializer, TokenizerModel *const model);

  const std::string &error_message() const;

private:
  bool SetError(const std::string &message);

private:
  std::string error_message_;
};

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
