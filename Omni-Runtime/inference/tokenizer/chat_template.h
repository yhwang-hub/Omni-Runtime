#pragma once

#include <string>

namespace omni_runtime {
namespace inference {

class Qwen3VLChatTemplate final {
public:
  bool BuildUserPrompt(const std::string &text, const bool has_image,
                       std::string *const prompt) const;
};

// Qwen3.5 keeps the ChatML frame of Qwen3-VL but always opens the assistant turn
// with a reasoning block, empty when thinking is disabled.
class Qwen3_5ChatTemplate final {
public:
  bool BuildUserPrompt(const std::string &text, const bool has_image, const bool enable_thinking,
                       std::string *const prompt) const;
};

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
