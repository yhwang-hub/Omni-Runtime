#include "Omni-Runtime/inference/tokenizer/chat_template.h"

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {

bool Qwen3VLChatTemplate::BuildUserPrompt(const std::string &text, const bool has_image,
                                          std::string *const prompt) const {
  OMNI_RETURN_VAL_IF(prompt == nullptr || text.empty(), false);
  *prompt = "<|im_start|>user\n";
  if (has_image) {
    *prompt += "<|vision_start|><|image_pad|><|vision_end|>\n";
  }
  *prompt += text;
  *prompt += "<|im_end|>\n<|im_start|>assistant\n";
  return true;
}

bool Qwen3_5ChatTemplate::BuildUserPrompt(const std::string &text, const bool has_image,
                                          const bool enable_thinking,
                                          std::string *const prompt) const {
  OMNI_RETURN_VAL_IF(prompt == nullptr || text.empty(), false);
  *prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n";
  if (has_image) {
    *prompt += "<|vision_start|><|image_pad|><|vision_end|>";
  }
  *prompt += text;
  *prompt += "<|im_end|>\n<|im_start|>assistant\n";
  *prompt += enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  return true;
}

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
