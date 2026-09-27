#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "Omni-Runtime/models/qwen3_8_nvfp4/qwen3_8_nvfp4_pipeline.h"

namespace omni_runtime::inference {
namespace {

std::filesystem::path RunfilePath(const std::string &relative_path) {
  const char *const root = std::getenv("TEST_SRCDIR");
  return root != nullptr && root[0] != '\0' ? std::filesystem::path(root) / "_main" / relative_path
                                            : std::filesystem::path(relative_path);
}

class Qwen38InferenceTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    pipeline_ = std::make_unique<qwen3_8_nvfp4::Qwen38Pipeline>();
    ASSERT_TRUE(
        pipeline_->Init(RunfilePath("Omni-Runtime/inference/conf/qwen3_8_27B_nvfp4.conf").string()))
        << pipeline_->error_message();
  }

  static void TearDownTestSuite() {
    pipeline_.reset();
  }

  static inline std::unique_ptr<qwen3_8_nvfp4::Qwen38Pipeline> pipeline_;
};

TEST_F(Qwen38InferenceTest, GreedyImageOutputMatchesLlamaCpp) {
  const auto &config = pipeline_->model_config();
  EXPECT_EQ(config.layer_count, 64);
  EXPECT_EQ(config.hidden_size, 5120);
  EXPECT_EQ(config.vocabulary_size, 248320);
  EXPECT_EQ(config.rotary_size, 64);
  GenerationRequest request;
  request.prompt = "Describe this image.";
  request.image_path = RunfilePath("test/inference/demo.jpeg").string();
  request.max_new_tokens = 512;
  std::string streamed;
  request.stream_callback = [&streamed](const std::string &chunk) {
    streamed += chunk;
    std::cout << chunk << std::flush;
    return true;
  };
  GenerationResult result;
  const auto start = std::chrono::steady_clock::now();
  ASSERT_TRUE(pipeline_->Generate(request, &result)) << pipeline_->error_message();
  const double wall_time_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  std::cout << std::fixed << std::setprecision(2) << "\n\nModel infer Time:\n"
            << "  Backend: " << result.backend_name << '\n'
            << "  VIT: " << result.vision_time_ms << " ms\n"
            << "  Prefill: " << result.prefill_time_ms << " ms, "
            << result.prefill_tokens_per_second() << " tokens/s\n"
            << "  Decode: " << result.decode_time_ms << " ms, " << result.decode_tokens_per_second()
            << " tokens/s\n"
            << "  Generation wall time (including graph preparation): " << wall_time_ms << " ms\n"
            << "  Generated tokens: " << result.generated_token_ids.size() << '\n';
  std::ifstream reference(RunfilePath("test/inference/qwen3_8_27b_llama.txt"), std::ios::binary);
  ASSERT_TRUE(reference.good());
  std::string expected{std::istreambuf_iterator<char>(reference), std::istreambuf_iterator<char>()};
  ASSERT_FALSE(expected.empty());
  ASSERT_EQ(expected.back(), '\n');
  expected.pop_back();
  EXPECT_EQ(result.backend_name, "native_" + pipeline_->compute_backend() + "(omni-runtime)");
  EXPECT_EQ(result.text, streamed);
  EXPECT_EQ(result.text, expected);
  EXPECT_EQ(result.prefill_token_count, 2766);
  EXPECT_EQ(result.vision_token_count, 2752);
  ASSERT_FALSE(result.generated_token_ids.empty());
  EXPECT_EQ(result.generated_token_ids.back(), config.end_of_sequence_id);
  EXPECT_EQ(result.decode_token_count + 1, result.generated_token_ids.size());
}

TEST_F(Qwen38InferenceTest, ResetsRecurrentStateAfterCancellation) {
  GenerationRequest request;
  request.prompt = "Describe this image.";
  request.max_new_tokens = 4;
  request.stream_callback = [](const std::string &) { return false; };
  GenerationResult first;
  ASSERT_TRUE(pipeline_->Generate(request, &first)) << pipeline_->error_message();
  ASSERT_EQ(first.generated_token_ids.size(), 1U);
  EXPECT_TRUE(first.is_cancelled);
  request.stream_callback = {};
  request.max_new_tokens = 1;
  GenerationResult second;
  ASSERT_TRUE(pipeline_->Generate(request, &second)) << pipeline_->error_message();
  EXPECT_EQ(first.generated_token_ids, second.generated_token_ids);
  EXPECT_EQ(first.text, second.text);
  EXPECT_FALSE(second.is_cancelled);
  EXPECT_EQ(first.gpu_memory_bytes, second.gpu_memory_bytes);
  EXPECT_EQ(second.decode_time_ms, 0.0);
}

TEST_F(Qwen38InferenceTest, RejectsInvalidRequests) {
  GenerationResult result;
  GenerationRequest request;
  EXPECT_FALSE(pipeline_->Generate(request, &result));
  request.prompt = "Hello";
  request.max_new_tokens = 1;
  EXPECT_FALSE(pipeline_->Generate(request, nullptr));
  request.image_path = "/nonexistent/omni-runtime-image.jpeg";
  EXPECT_FALSE(pipeline_->Generate(request, &result));
  request.image_path.clear();
  request.max_new_tokens = 8192;
  EXPECT_FALSE(pipeline_->Generate(request, &result));
}

} // namespace
} // namespace omni_runtime::inference
