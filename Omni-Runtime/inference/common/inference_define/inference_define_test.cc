#include "Omni-Runtime/inference/common/inference_define/inference_define.h"

#include "gtest/gtest.h"

namespace omni_runtime::inference {

TEST(GenerationResultTest, DefaultsToEmptyImageAndUnmeasuredThroughput) {
  const GenerationResult result;
  EXPECT_EQ(result.input_image_size.width_pixels, 0);
  EXPECT_EQ(result.input_image_size.height_pixels, 0);
  EXPECT_EQ(result.resized_image_size.width_pixels, 0);
  EXPECT_EQ(result.resized_image_size.height_pixels, 0);
  EXPECT_EQ(result.decode_token_count, 0);
  EXPECT_DOUBLE_EQ(result.prefill_tokens_per_second(), 0.0);
  EXPECT_DOUBLE_EQ(result.decode_tokens_per_second(), 0.0);
}

TEST(GenerationResultTest, ComputesRatesFromEachTimedStagesTokenCount) {
  GenerationResult result;
  result.prefill_token_count = 2766;
  result.prefill_time_ms = 500.0;
  result.generated_token_ids.resize(352U);
  result.decode_token_count = 351;
  result.decode_time_ms = 20000.0;
  EXPECT_DOUBLE_EQ(result.prefill_tokens_per_second(), 5532.0);
  EXPECT_DOUBLE_EQ(result.decode_tokens_per_second(), 17.55);
}

TEST(GenerationResultTest, HandlesZeroDurationWithoutDividingByZero) {
  GenerationResult result;
  result.prefill_token_count = 1;
  result.decode_token_count = 1;
  EXPECT_DOUBLE_EQ(result.prefill_tokens_per_second(), 0.0);
  EXPECT_DOUBLE_EQ(result.decode_tokens_per_second(), 0.0);
}

} // namespace omni_runtime::inference
