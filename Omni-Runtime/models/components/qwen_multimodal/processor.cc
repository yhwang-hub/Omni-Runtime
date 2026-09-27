#include "Omni-Runtime/models/components/qwen_multimodal/processor.h"

#include <algorithm>

#include "Omni-Runtime/inference/tokenizer/chat_template.h"
#include "Omni-Runtime/inference/tokenizer/gguf_tokenizer_loader.h"
#include "Omni-Runtime/inference/tokenizer/tokenizer_factory.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3 {

bool QwenProcessor::Init(const util::GGUFSerializer &serializer) {
  tokenizer_.reset();
  error_message_.clear();
  GGUFTokenizerLoader loader;
  TokenizerModel model;
  OMNI_RETURN_VAL_IF(!loader.Load(serializer, &model),
                     (error_message_ = loader.error_message(), false));
  return Init(model);
}

bool QwenProcessor::Init(const TokenizerModel &model) {
  tokenizer_.reset();
  error_message_.clear();
  auto tokenizer = TokenizerFactory::Create(model.type);
  OMNI_RETURN_VAL_IF(tokenizer == nullptr, (error_message_ = "failed to create tokenizer", false));
  OMNI_RETURN_VAL_IF(!tokenizer->Init(model), (error_message_ = tokenizer->error_message(), false));
  tokenizer_ = std::move(tokenizer);
  return true;
}

bool QwenProcessor::Prepare(const GenerationRequest &request, const int32_t image_id,
                            const int32_t vocabulary_size, const int32_t max_context_tokens,
                            Prompt *const prompt, std::string *const error_message) const {
  OMNI_RETURN_VAL_IF_LOG(prompt == nullptr || error_message == nullptr, false, ERROR,
                         "Prompt outputs are null");
  OMNI_RETURN_VAL_IF(tokenizer_ == nullptr, (*error_message = "prompt tokenizer is null", false));
  error_message->clear();
  std::string text;
  const Qwen3VLChatTemplate chat;
  OMNI_RETURN_VAL_IF(
      !chat.BuildUserPrompt(request.prompt, !request.image_path.empty(), &text) ||
          !tokenizer_->Encode(text, &prompt->tokens),
      (*error_message = "prompt tokenization failed: " + tokenizer_->error_message(), false));
  return Expand(image_id, vocabulary_size, max_context_tokens, request.max_new_tokens, prompt,
                error_message);
}

bool QwenProcessor::Expand(const int32_t image_id, const int32_t vocabulary_size,
                           const int32_t max_context_tokens, const int32_t max_new_tokens,
                           Prompt *const prompt, std::string *const error_message) {
  OMNI_RETURN_VAL_IF_LOG(prompt == nullptr || error_message == nullptr, false, ERROR,
                         "Prompt outputs are null");
  error_message->clear();
  OMNI_RETURN_VAL_IF(vocabulary_size <= 0 || max_context_tokens <= 0 || max_new_tokens <= 0 ||
                         prompt->image_tokens < 0,
                     (*error_message = "invalid prompt dimensions or generation budget", false));
  OMNI_RETURN_VAL_IF(prompt->tokens.empty() ||
                         prompt->tokens.size() + static_cast<size_t>(max_new_tokens) >
                             static_cast<size_t>(max_context_tokens),
                     (*error_message = "prompt and generation exceed context capacity", false));
  prompt->image_offset = -1;
  if (prompt->image_tokens > 0) {
    const auto image = std::find(prompt->tokens.begin(), prompt->tokens.end(), image_id);
    OMNI_RETURN_VAL_IF(
        image == prompt->tokens.end() || prompt->image_grid_width <= 0 ||
            prompt->image_grid_height <= 0 ||
            static_cast<int64_t>(prompt->image_grid_height) * prompt->image_grid_width !=
                prompt->image_tokens,
        (*error_message = "image placeholder or vision features are invalid", false));
    OMNI_RETURN_VAL_IF(prompt->tokens.size() + static_cast<size_t>(prompt->image_tokens - 1) +
                               static_cast<size_t>(max_new_tokens) >
                           static_cast<size_t>(max_context_tokens),
                       (*error_message = "prompt and generation exceed context capacity", false));
    prompt->image_offset = static_cast<int32_t>(std::distance(prompt->tokens.begin(), image));
    prompt->tokens.insert(image, prompt->image_tokens - 1, image_id);
  }
  OMNI_RETURN_VAL_IF(
      std::any_of(prompt->tokens.begin(), prompt->tokens.end(),
                  [vocabulary_size](const int32_t id) { return id < 0 || id >= vocabulary_size; }),
      (*error_message = "token ID is outside the vocabulary", false));
  const int32_t count = static_cast<int32_t>(prompt->tokens.size());
  prompt->positions.resize(static_cast<size_t>(count) * 3U);
  int32_t position = 0;
  for (int32_t token = 0; token < count; ++token) {
    if (prompt->image_offset >= 0 && token >= prompt->image_offset &&
        token < prompt->image_offset + prompt->image_tokens) {
      const int32_t image = token - prompt->image_offset;
      prompt->positions[token] = prompt->image_offset;
      prompt->positions[static_cast<size_t>(count) + token] =
          prompt->image_offset + image / prompt->image_grid_width;
      prompt->positions[static_cast<size_t>(count) * 2U + token] =
          prompt->image_offset + image % prompt->image_grid_width;
      position =
          prompt->image_offset + std::max(prompt->image_grid_height, prompt->image_grid_width);
    } else {
      for (int32_t axis = 0; axis < 3; ++axis) {
        prompt->positions[static_cast<size_t>(axis) * count + token] = position;
      }
      ++position;
    }
  }
  prompt->next_position = position;
  return true;
}

const Tokenizer *QwenProcessor::tokenizer() const {
  return tokenizer_.get();
}

const std::string &QwenProcessor::error_message() const {
  return error_message_;
}

} // namespace qwen3
} // namespace inference
} // namespace omni_runtime
