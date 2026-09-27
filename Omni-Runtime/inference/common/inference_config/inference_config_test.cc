#include <gtest/gtest.h>

#include "Omni-Runtime/inference/common/inference_config/inference_config.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

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
  EXPECT_EQ(config.graph_path(), "qwen3_vl.yaml");
  std::string graph_path;
  ASSERT_TRUE(config.ResolveRelativePath(config.graph_path(), &graph_path));
  EXPECT_EQ(std::filesystem::path(graph_path).filename(), "qwen3_vl.yaml");
}

class InferenceConfigPathTest : public ::testing::Test {
protected:
  void SetUp() final {
    directory_ = std::filesystem::path(::testing::TempDir()) /
                 (std::string("inference_config_") +
                  ::testing::UnitTest::GetInstance()->current_test_info()->name());
    ASSERT_TRUE(std::filesystem::create_directory(directory_));
    config_path_ = directory_ / "runtime.conf";
    graph_path_ = directory_ / "graph.yaml";
    std::ofstream graph(graph_path_);
    graph << "backend: cuda\n";
    ASSERT_TRUE(graph.good());
  }

  void TearDown() final {
    for (const auto &path : {config_path_, graph_path_, directory_}) {
      std::error_code error;
      std::filesystem::remove(path, error);
      EXPECT_FALSE(error) << path << ": " << error.message();
    }
  }

  bool WriteConfig(const std::string &contents) const {
    std::ofstream output(config_path_);
    output << contents;
    output.close();
    return !output.fail();
  }

  std::filesystem::path directory_;
  std::filesystem::path config_path_;
  std::filesystem::path graph_path_;
};

TEST_F(InferenceConfigPathTest, ResolvesGraphRelativeToConfigOutsideWorkingDirectory) {
  ASSERT_TRUE(WriteConfig("model_name=qwen3_vl\ngraph_path=graph.yaml\n"));
  InferenceConfig config;
  ASSERT_TRUE(config.Load(config_path_.string())) << config.error_message();
  EXPECT_EQ(config.graph_path(), "graph.yaml");
  std::string resolved_path;
  ASSERT_TRUE(config.ResolveRelativePath(config.graph_path(), &resolved_path));
  EXPECT_EQ(std::filesystem::path(resolved_path), graph_path_);
}

TEST_F(InferenceConfigPathTest, ResolvesAbsoluteGraphPath) {
  ASSERT_TRUE(WriteConfig("model_name=qwen3_vl\ngraph_path=" + graph_path_.string() + "\n"));
  InferenceConfig config;
  ASSERT_TRUE(config.Load(config_path_.string())) << config.error_message();
  std::string resolved_path;
  ASSERT_TRUE(config.ResolveRelativePath(config.graph_path(), &resolved_path));
  EXPECT_EQ(std::filesystem::path(resolved_path), graph_path_);
}

TEST_F(InferenceConfigPathTest, PreservesRepositoryRelativeGraphPathSupport) {
  const char *const test_srcdir = std::getenv("TEST_SRCDIR");
  const auto root = test_srcdir == nullptr ? std::filesystem::current_path()
                                           : std::filesystem::path(test_srcdir) / "_main";
  const std::string relative_path = "Omni-Runtime/inference/conf/qwen3_vl.yaml";
  InferenceConfig config;
  ASSERT_TRUE(config.Load((root / "Omni-Runtime/inference/conf/qwen3_vl_8B_f16.conf").string()));
  std::string resolved_path;
  ASSERT_TRUE(config.ResolveRelativePath(relative_path, &resolved_path));
  EXPECT_EQ(std::filesystem::path(resolved_path), root / relative_path);
}

TEST_F(InferenceConfigPathTest, RejectsMissingGraphAndNullPathOutput) {
  ASSERT_TRUE(WriteConfig("model_name=qwen3_vl\ngraph_path=missing.yaml\n"));
  InferenceConfig config;
  ASSERT_TRUE(config.Load(config_path_.string()));
  std::string resolved_path;
  EXPECT_FALSE(config.ResolveRelativePath(config.graph_path(), &resolved_path));
  EXPECT_FALSE(config.ResolveRelativePath(config.graph_path(), nullptr));
}

TEST_F(InferenceConfigPathTest, ClearsGraphPathWhenReloadingConfigWithoutGraph) {
  ASSERT_TRUE(WriteConfig("model_name=qwen3_vl\ngraph_path=graph.yaml\n"));
  InferenceConfig config;
  ASSERT_TRUE(config.Load(config_path_.string()));
  EXPECT_FALSE(config.graph_path().empty());
  ASSERT_TRUE(WriteConfig("model_name=pi0_5\n"));
  ASSERT_TRUE(config.Load(config_path_.string()));
  EXPECT_TRUE(config.graph_path().empty());
}

TEST_F(InferenceConfigPathTest, RejectsEmptyGraphValue) {
  ASSERT_TRUE(WriteConfig("model_name=qwen3_vl\ngraph_path=  \n"));
  InferenceConfig config;
  EXPECT_FALSE(config.Load(config_path_.string()));
  EXPECT_FALSE(config.error_message().empty());
}

} // namespace
} // namespace inference
} // namespace omni_runtime
