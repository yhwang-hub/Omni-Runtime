#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "gtest/gtest.h"

#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_pipeline.h"

namespace {

std::filesystem::path RunfilePath(const std::string &relative_path) {
  const char *const test_srcdir = std::getenv("TEST_SRCDIR");
  if (test_srcdir != nullptr && test_srcdir[0] != '\0') {
    return std::filesystem::path(test_srcdir) / "_main" / relative_path;
  }
  return std::filesystem::path(relative_path);
}

bool ReadReferenceOutput(std::string *const output) {
  if (output == nullptr) {
    return false;
  }
  const std::filesystem::path path = RunfilePath("test/inference/qwen3_vl_8b_expected.txt");
  std::ifstream stream(path, std::ios::binary);
  if (!stream.good()) {
    return false;
  }
  output->assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
  return stream.good() || stream.eof();
}

} // namespace

namespace omni_runtime {
namespace inference {

TEST(Qwen3VLInferenceTest, MatchesModelScopeReferenceOutput) {
  qwen3_vl::Qwen3VLPipeline pipeline;
  ASSERT_TRUE(
      pipeline.Init(RunfilePath("Omni-Runtime/inference/conf/qwen3_vl_8B_f16.conf").string()));
  std::string actual_output;
  ASSERT_TRUE(pipeline.Run(&actual_output));
  std::string expected_output;
  ASSERT_TRUE(ReadReferenceOutput(&expected_output));
  std::cout << actual_output << std::endl;
  EXPECT_EQ(actual_output, expected_output);
}

} // namespace inference
} // namespace omni_runtime
