#pragma once

#include <memory>
#include <string>

#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"
#include "Omni-Runtime/inference/tokenizer/tokenizer.h"
#include "Omni-Runtime/inference/common/inference_define/inference_define.h"
#include "Omni-Runtime/models/components/qwen_multimodal/processor_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3 {

class QwenProcessor final {
public:
  bool Init(const util::GGUFSerializer &serializer);
  bool Init(const TokenizerModel &model);
  bool Prepare(const GenerationRequest &request, const int32_t image_id,
               const int32_t vocabulary_size, const int32_t max_context_tokens,
               Prompt *const prompt, std::string *const error_message) const;
  const Tokenizer *tokenizer() const;
  const std::string &error_message() const;

  static bool Expand(const int32_t image_id, const int32_t vocabulary_size,
                     const int32_t max_context_tokens, const int32_t max_new_tokens,
                     Prompt *const prompt, std::string *const error_message);

private:
  std::unique_ptr<Tokenizer> tokenizer_;
  std::string error_message_;
};

} // namespace qwen3
} // namespace inference
} // namespace omni_runtime
