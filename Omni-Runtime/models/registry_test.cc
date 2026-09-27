#include "Omni-Runtime/models/registry.h"

#include "gtest/gtest.h"

namespace omni_runtime {
namespace inference {
namespace {

TEST(ModelRegistryTest, ResolvesArchitectureAndDeploymentAliasesWithoutLoadingWeights) {
  ModelRegistry registry;
  ASSERT_TRUE(RegisterBuiltinModels(&registry));
  for (const std::string name : {"qwen3_vl", "qwen3vl", "qwen3_8_nvfp4", "qwen3_5", "qwen35"}) {
    std::unique_ptr<Model> model;
    ASSERT_TRUE(registry.Create(name, &model)) << registry.error_message();
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->descriptor().architecture,
              name == "qwen3_vl" || name == "qwen3vl" ? "qwen3vl" : "qwen35");
    EXPECT_EQ(model->descriptor().task, ModelTask::kTextGeneration);
    EXPECT_TRUE(model->descriptor().supports_images);
  }
}

TEST(ModelRegistryTest, RejectsUnknownDuplicateEmptyAndNullFactories) {
  ModelRegistry registry;
  ASSERT_TRUE(RegisterBuiltinModels(&registry));
  std::unique_ptr<Model> model;
  ASSERT_TRUE(registry.Create("qwen3_vl", &model));
  const Model *const original = model.get();
  EXPECT_FALSE(registry.Create("unknown", &model));
  EXPECT_EQ(model.get(), original);
  EXPECT_FALSE(registry.Create("qwen3_vl", nullptr));
  EXPECT_FALSE(RegisterBuiltinModels(&registry));
  EXPECT_FALSE(RegisterBuiltinModels(nullptr));
  EXPECT_FALSE(registry.Register("", ModelRegistry::Factory{}));
  EXPECT_FALSE(registry.Register("empty", ModelRegistry::Factory{}));
  ASSERT_TRUE(registry.Register("null", []() { return std::unique_ptr<Model>{}; }));
  EXPECT_FALSE(registry.Create("null", &model));
  EXPECT_EQ(model.get(), original);
}

TEST(ModelRegistryTest, CreatesFromConfigWithoutInitializingDeviceStorage) {
  ModelRegistry registry;
  ASSERT_TRUE(RegisterBuiltinModels(&registry));
  std::unique_ptr<Model> model;
  ASSERT_TRUE(
      registry.CreateFromConfig("Omni-Runtime/inference/conf/qwen3_8_27B_nvfp4.conf", &model))
      << registry.error_message();
  EXPECT_EQ(model->descriptor().architecture, "qwen35");
  EXPECT_FALSE(registry.CreateFromConfig("missing.conf", &model));
}

} // namespace
} // namespace inference
} // namespace omni_runtime
