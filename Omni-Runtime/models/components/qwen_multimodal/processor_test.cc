#include "Omni-Runtime/models/components/qwen_multimodal/processor.h"

#include <limits>

#include "gtest/gtest.h"

namespace omni_runtime {
namespace inference {
namespace qwen3 {
namespace {

TEST(QwenProcessorTest, ExpandsImageTokensAndPreservesThreeAxisPositions) {
  Prompt prompt;
  prompt.tokens = {1, 99, 2, 3};
  prompt.image_tokens = 4;
  prompt.image_grid_height = 2;
  prompt.image_grid_width = 2;
  std::string error_message;
  ASSERT_TRUE(QwenProcessor::Expand(99, 100, 16384, 8, &prompt, &error_message));
  EXPECT_EQ(prompt.tokens, (std::vector<int32_t>{1, 99, 99, 99, 99, 2, 3}));
  EXPECT_EQ(prompt.positions,
            (std::vector<int32_t>{0, 1, 1, 1, 1, 3, 4, 0, 1, 1, 2, 2, 3, 4, 0, 1, 2, 1, 2, 3, 4}));
  EXPECT_EQ(prompt.next_position, 5);
  prompt = Prompt{};
  prompt.tokens = {1, 2, 3};
  ASSERT_TRUE(QwenProcessor::Expand(99, 100, 16384, 8, &prompt, &error_message));
  EXPECT_EQ(prompt.positions, (std::vector<int32_t>{0, 1, 2, 0, 1, 2, 0, 1, 2}));
  EXPECT_EQ(prompt.image_offset, -1);
}

TEST(QwenProcessorTest, RejectsInvalidSpansAndChecksCapacityBeforeExpansion) {
  Prompt prompt;
  prompt.tokens = {1, 2};
  prompt.image_tokens = 4;
  prompt.image_grid_height = 2;
  prompt.image_grid_width = 2;
  std::string error_message;
  EXPECT_FALSE(QwenProcessor::Expand(99, 100, 16384, 8, &prompt, &error_message));
  prompt.tokens = {99};
  prompt.image_grid_height = 0;
  EXPECT_FALSE(QwenProcessor::Expand(99, 100, 16384, 8, &prompt, &error_message));
  prompt.image_grid_height = 2;
  prompt.image_tokens = std::numeric_limits<int32_t>::max();
  EXPECT_FALSE(QwenProcessor::Expand(99, 100, 16384, 8, &prompt, &error_message));
  EXPECT_EQ(prompt.tokens.size(), 1U);
  prompt = Prompt{};
  prompt.tokens = {-1};
  EXPECT_FALSE(QwenProcessor::Expand(99, 100, 16384, 8, &prompt, &error_message));
  prompt.tokens = {100};
  EXPECT_FALSE(QwenProcessor::Expand(99, 100, 16384, 8, &prompt, &error_message));
  prompt.tokens = {1};
  EXPECT_FALSE(QwenProcessor::Expand(99, 100, 16384, 16384, &prompt, &error_message));
  EXPECT_NE(error_message.find("context capacity"), std::string::npos);
  ASSERT_TRUE(QwenProcessor::Expand(99, 100, 16384, 16383, &prompt, &error_message));
  EXPECT_TRUE(error_message.empty());
  EXPECT_FALSE(QwenProcessor::Expand(99, 100, 16384, 8, nullptr, &error_message));
  EXPECT_FALSE(QwenProcessor::Expand(99, 100, 16384, 8, &prompt, nullptr));
}

TEST(QwenProcessorTest, RejectsUseBeforeInitializationAndMalformedTokenizer) {
  QwenProcessor processor;
  GenerationRequest request;
  request.prompt = "Hello";
  request.max_new_tokens = 1;
  Prompt prompt;
  std::string error_message;
  EXPECT_FALSE(processor.Prepare(request, 99, 100, 16384, &prompt, &error_message));
  EXPECT_FALSE(error_message.empty());
  const TokenizerModel model;
  EXPECT_FALSE(processor.Init(model));
  EXPECT_EQ(processor.tokenizer(), nullptr);
}

} // namespace
} // namespace qwen3
} // namespace inference
} // namespace omni_runtime
