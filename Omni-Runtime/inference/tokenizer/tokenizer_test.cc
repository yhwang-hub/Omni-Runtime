#include "Omni-Runtime/inference/tokenizer/tokenizer_factory.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "Omni-Runtime/inference/tokenizer/chat_template.h"
#include "Omni-Runtime/inference/tokenizer/gguf_tokenizer_loader.h"
#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"

namespace omni_runtime {
namespace inference {
namespace {

std::string ModelPath() {
  const char *const model_directory = std::getenv("QWEN3_VL_GGUF_MODEL_DIR");
  const std::string directory = model_directory == nullptr ? "model/qwen3_vl" : model_directory;
  return directory + "/Qwen3VL-8B-Instruct-F16.gguf";
}

TEST(TokenizerTest, EncodesQwen3VLImagePromptExactly) {
  util::GGUFSerializer serializer;
  ASSERT_TRUE(serializer.Parse(ModelPath())) << serializer.error_message();
  TokenizerModel model;
  GGUFTokenizerLoader loader;
  ASSERT_TRUE(loader.Load(serializer, &model)) << loader.error_message();
  std::unique_ptr<Tokenizer> tokenizer = TokenizerFactory::Create(model.type);
  ASSERT_NE(tokenizer, nullptr);
  ASSERT_TRUE(tokenizer->Init(model)) << tokenizer->error_message();

  std::string prompt;
  Qwen3VLChatTemplate chat_template;
  ASSERT_TRUE(chat_template.BuildUserPrompt("Describe this image.", true, &prompt));
  const std::vector<int32_t> expected_token_ids = {151644, 872,    198,   151652, 151655, 151653,
                                                   198,    74785,  419,   2168,   13,     151645,
                                                   198,    151644, 77091, 198};
  std::vector<int32_t> token_ids;
  ASSERT_TRUE(tokenizer->Encode(prompt, &token_ids)) << tokenizer->error_message();
  EXPECT_EQ(token_ids, expected_token_ids);

  std::string decoded_text;
  ASSERT_TRUE(tokenizer->Decode(token_ids, &decoded_text)) << tokenizer->error_message();
  EXPECT_EQ(decoded_text, prompt);
}

TEST(TokenizerTest, FactorySupportsWordPieceAndUnigram) {
  TokenizerModel word_piece_model;
  word_piece_model.type = TokenizerType::WORD_PIECE;
  word_piece_model.vocabulary = {"[UNK]", "hello", "##s", "world"};
  word_piece_model.unknown_token_id = 0;
  std::unique_ptr<Tokenizer> word_piece = TokenizerFactory::Create(TokenizerType::WORD_PIECE);
  ASSERT_NE(word_piece, nullptr);
  ASSERT_TRUE(word_piece->Init(word_piece_model));
  std::vector<int32_t> token_ids;
  ASSERT_TRUE(word_piece->Encode("hellos world", &token_ids));
  EXPECT_EQ(token_ids, std::vector<int32_t>({1, 2, 3}));

  TokenizerModel unigram_model;
  unigram_model.type = TokenizerType::UNIGRAM;
  unigram_model.vocabulary = {"a", "b", "ab"};
  unigram_model.scores = {-1.0f, -1.0f, -0.5f};
  std::unique_ptr<Tokenizer> unigram = TokenizerFactory::Create(TokenizerType::UNIGRAM);
  ASSERT_NE(unigram, nullptr);
  ASSERT_TRUE(unigram->Init(unigram_model));
  ASSERT_TRUE(unigram->Encode("ab", &token_ids));
  EXPECT_EQ(token_ids, std::vector<int32_t>({2}));
}

} // namespace
} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
