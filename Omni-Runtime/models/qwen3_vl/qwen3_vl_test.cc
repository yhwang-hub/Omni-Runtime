#include "gtest/gtest.h"

#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_runtime.h"

namespace omni_runtime::inference::qwen3_vl {
namespace {

TEST(Qwen3VLModelTest, ModelLibraryLoads) {
  SUCCEED();
}

TEST(Qwen3VLModelTest, PreservesPreallocatedStorageAcrossGenerationResets) {
  Qwen3VLRuntime runtime;
  runtime.backend = backend::CreateBackend(backend::BackendKind::kCuda);
  if (runtime.backend == nullptr || !runtime.backend->IsAvailable()) {
    GTEST_SKIP() << "GPU backend is unavailable";
  }
  const inference::InferenceConfig config;
  runtime.config = &config;
  runtime.model_config.layer_count = 1;
  runtime.model_config.hidden_size = 4;
  runtime.model_config.feed_forward_size = 8;
  runtime.model_config.query_head_count = 2;
  runtime.model_config.key_value_head_count = 1;
  runtime.model_config.head_size = 2;
  runtime.model_config.vocabulary_size = 16;
  runtime.vision_config.hidden_size = 4;
  runtime.vision_config.feed_forward_size = 8;
  runtime.vision_config.patch_size = 1;
  runtime.vision_config.merge_size = 2;
  runtime.vision_config.projection_size = 4;
  runtime.vision_config.deepstack_layers = {0};
  ASSERT_TRUE(AllocateKvCache(8, &runtime));
  ASSERT_TRUE(AllocateWorkspaces(&runtime));
  const auto key_cache = runtime.key_cache;
  const auto value_cache = runtime.value_cache;
  const auto language_hidden = runtime.language_workspaces.hidden;
  const auto vision_hidden = runtime.vision_workspaces.hidden;
  const auto vision_embedding = runtime.vision_workspaces.embedding;
  const auto deepstack = runtime.vision_workspaces.deepstack;
  const size_t gpu_memory_bytes = runtime.gpu_memory_bytes;
  runtime.used_token_count = 4;
  runtime.next_rope_position = 2;
  ASSERT_TRUE(ResetGenerationState(&runtime));
  EXPECT_EQ(runtime.used_token_count, 0);
  EXPECT_EQ(runtime.next_rope_position, 0);
  EXPECT_EQ(runtime.cache_capacity, 8);
  EXPECT_EQ(runtime.key_cache, key_cache);
  EXPECT_EQ(runtime.value_cache, value_cache);
  EXPECT_EQ(runtime.language_workspaces.hidden, language_hidden);
  EXPECT_EQ(runtime.vision_workspaces.hidden, vision_hidden);
  EXPECT_EQ(runtime.vision_workspaces.embedding, vision_embedding);
  EXPECT_EQ(runtime.vision_workspaces.deepstack, deepstack);
  EXPECT_EQ(runtime.gpu_memory_bytes, gpu_memory_bytes);
}

} // namespace
} // namespace omni_runtime::inference::qwen3_vl
