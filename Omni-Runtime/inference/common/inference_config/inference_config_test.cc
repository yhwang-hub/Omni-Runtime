#include <gtest/gtest.h>

#include "Omni-Runtime/inference/common/inference_config/inference_config.h"

#include <cstdlib>

namespace omni_runtime {
namespace inference {
namespace {

TEST(InferenceConfigTest, LoadsQwen3VlRuntimeConfig) {
  InferenceConfig config;
  const char *const test_srcdir = std::getenv("TEST_SRCDIR");
  const std::string path =
      test_srcdir == nullptr
          ? "Omni-Runtime/inference/conf/qwen3_vl_8B_f16.conf"
          : std::string(test_srcdir) + "/_main/Omni-Runtime/inference/conf/qwen3_vl_8B_f16.conf";
  ASSERT_TRUE(config.Load(path)) << config.error_message();
  EXPECT_EQ(config.model_name(), "qwen3_vl");
  EXPECT_EQ(config.model_backend(), ModelBackend::kGpu);
  EXPECT_EQ(config.model_weight_type(), ModelWeightType::kFloat16);
  EXPECT_EQ(config.compute_type(), ModelWeightType::kFloat16);
  EXPECT_EQ(config.max_new_tokens(), 1024);
}

} // namespace
} // namespace inference
} // namespace omni_runtime
