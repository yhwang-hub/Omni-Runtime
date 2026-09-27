#include "Omni-Runtime/backend/backend.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace omni_runtime {
namespace backend {
namespace {

class ComputeGraphTest : public ::testing::Test {
protected:
  DeviceView view(const size_t offset_elements, const size_t count_elements) {
    return {.data = storage_.data(),
            .offset_bytes = offset_elements * sizeof(float),
            .size_bytes = count_elements * sizeof(float),
            .backend_kind = BackendKind::kCuda};
  }

  ParallelLinear projections() {
    return {.projections = {{.input = view(0, 4),
                             .weight = view(4, 8),
                             .rows = 1,
                             .inner = 4,
                             .columns = 2,
                             .output = view(48, 2)},
                            {.input = view(0, 4),
                             .weight = view(12, 12),
                             .rows = 1,
                             .inner = 4,
                             .columns = 3,
                             .output = view(50, 3)},
                            {.input = view(0, 4),
                             .weight = view(24, 16),
                             .rows = 1,
                             .inner = 4,
                             .columns = 4,
                             .output = view(53, 4)}}};
  }

  std::array<float, 128U> storage_ = {};
};

TEST_F(ComputeGraphTest, LowersIndependentProjectionsAndPreservesJoin) {
  const auto group = projections();
  const std::vector<ComputeNode> nodes = {{.operation = group.projections[0], .dependencies = {}},
                                          {.operation = group.projections[1], .dependencies = {}},
                                          {.operation = group.projections[2], .dependencies = {}},
                                          {.operation = Add{.lhs = group.projections[0].output,
                                                            .rhs = group.projections[1].output,
                                                            .element_count = 2U,
                                                            .output = view(60, 2)},
                                           .dependencies = {0U, 1U, 2U}}};
  std::vector<ComputeOperation> operations;
  ASSERT_TRUE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kCuda, nodes, &operations));
  ASSERT_EQ(operations.size(), 2U);
  const auto *const parallel = std::get_if<ParallelLinear>(&operations.front());
  ASSERT_NE(parallel, nullptr);
  EXPECT_EQ(parallel->projections.size(), 3U);
  EXPECT_TRUE(std::holds_alternative<Add>(operations.back()));
}

TEST_F(ComputeGraphTest, DoesNotGroupDependentOrDifferentlyShapedProjections) {
  auto group = projections();
  const std::vector<ComputeNode> dependent = {
      {.operation = group.projections[0], .dependencies = {}},
      {.operation = group.projections[1], .dependencies = {0U}},
      {.operation = group.projections[2], .dependencies = {1U}}};
  std::vector<ComputeOperation> operations;
  ASSERT_TRUE(
      CompileComputeGraph(ScalarType::kFloat32, BackendKind::kCuda, dependent, &operations));
  ASSERT_EQ(operations.size(), 3U);
  for (const auto &operation : operations) {
    EXPECT_TRUE(std::holds_alternative<Linear>(operation));
  }
  group.projections[1].input = view(80, 4);
  const std::vector<ComputeNode> distinct = {
      {.operation = group.projections[0], .dependencies = {}},
      {.operation = group.projections[1], .dependencies = {}}};
  ASSERT_TRUE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kCuda, distinct, &operations));
  EXPECT_EQ(operations.size(), 2U);
}

TEST_F(ComputeGraphTest, AcceptsForwardDependenciesAndTransitiveOrdering) {
  const auto group = projections();
  const std::vector<ComputeNode> nodes = {
      {.operation =
           Add{.lhs = view(48, 2), .rhs = view(48, 2), .element_count = 2U, .output = view(48, 2)},
       .dependencies = {2U}},
      {.operation = group.projections[0], .dependencies = {}},
      {.operation =
           Add{.lhs = view(48, 2), .rhs = view(48, 2), .element_count = 2U, .output = view(60, 2)},
       .dependencies = {1U}}};
  std::vector<ComputeOperation> operations;
  ASSERT_TRUE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kCuda, nodes, &operations));
  ASSERT_EQ(operations.size(), 3U);
  EXPECT_TRUE(std::holds_alternative<Linear>(operations.front()));
}

TEST_F(ComputeGraphTest, RejectsCyclesInvalidEdgesAndNullOutputWithoutPublishing) {
  const Linear projection = projections().projections[0];
  std::vector<ComputeOperation> operations = {AdvanceSequence{}};
  EXPECT_FALSE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kCuda, {}, &operations));
  EXPECT_FALSE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kCuda,
                                   {{.operation = projection, .dependencies = {0U}}}, &operations));
  EXPECT_FALSE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kCuda,
                                   {{.operation = projection, .dependencies = {2U}}}, &operations));
  EXPECT_FALSE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kCuda,
                                   {{.operation = projection, .dependencies = {1U}},
                                    {.operation = projection, .dependencies = {0U}}},
                                   &operations));
  EXPECT_FALSE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kCuda,
                                   {{.operation = projection, .dependencies = {}}}, nullptr));
  ASSERT_EQ(operations.size(), 1U);
  EXPECT_TRUE(std::holds_alternative<AdvanceSequence>(operations.front()));
}

TEST_F(ComputeGraphTest, RejectsUnorderedReadWriteAndWriteWriteAliases) {
  const auto group = projections();
  std::vector<ComputeOperation> operations;
  const ComputeOperation writes = group.projections[0];
  const ComputeOperation reads =
      Add{.lhs = view(49, 1), .rhs = view(50, 1), .element_count = 1U, .output = view(70, 1)};
  EXPECT_FALSE(CompileComputeGraph(
      ScalarType::kFloat32, BackendKind::kCuda,
      {{.operation = writes, .dependencies = {}}, {.operation = reads, .dependencies = {}}},
      &operations));
  EXPECT_FALSE(CompileComputeGraph(
      ScalarType::kFloat32, BackendKind::kCuda,
      {{.operation = reads, .dependencies = {}}, {.operation = writes, .dependencies = {}}},
      &operations));
  auto conflicting = group;
  conflicting.projections[1].output = view(49, 3);
  EXPECT_FALSE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kCuda,
                                   {{.operation = conflicting.projections[0], .dependencies = {}},
                                    {.operation = conflicting.projections[1], .dependencies = {}}},
                                   &operations));
  conflicting.projections[1].output = {.data = storage_.data() + 49,
                                       .offset_bytes = 0U,
                                       .size_bytes = 3U * sizeof(float),
                                       .backend_kind = BackendKind::kCuda};
  EXPECT_FALSE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kCuda, conflicting));
}

TEST_F(ComputeGraphTest, RejectsInvalidGroupsBeforeSubmission) {
  const auto original = projections();
  ASSERT_TRUE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kCuda, original));
  auto group = original;
  group.projections.resize(1U);
  EXPECT_FALSE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kCuda, group));
  group = original;
  group.projections.push_back(original.projections[0]);
  EXPECT_FALSE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kCuda, group));
  group = original;
  group.projections[1].rows = 0;
  EXPECT_FALSE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kCuda, group));
  group = original;
  group.projections[1].input = view(80, 4);
  EXPECT_FALSE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kCuda, group));
  group = original;
  group.projections[1].weight.size_bytes = 1U;
  EXPECT_FALSE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kCuda, group));
  group = original;
  group.projections[1].output = group.projections[0].input;
  EXPECT_FALSE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kCuda, group));
  group = original;
  group.projections[1].output = view(4, 3);
  EXPECT_FALSE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kCuda, group));
  EXPECT_FALSE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kOpenCL, original));
  EXPECT_FALSE(ValidateParallelLinear(static_cast<ScalarType>(255), BackendKind::kCuda, original));
}

TEST_F(ComputeGraphTest, ValidatesOpenClHandleOffsetsAndGroupDependencies) {
  auto group = projections();
  for (Linear &projection : group.projections) {
    projection.input.backend_kind = BackendKind::kOpenCL;
    projection.weight.backend_kind = BackendKind::kOpenCL;
    projection.output.backend_kind = BackendKind::kOpenCL;
  }
  EXPECT_TRUE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kOpenCL, group));
  std::vector<ComputeOperation> operations;
  EXPECT_TRUE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kOpenCL,
                                  {{.operation = group, .dependencies = {}}}, &operations));
  group.projections[1].output.offset_bytes = group.projections[0].output.offset_bytes;
  EXPECT_FALSE(ValidateParallelLinear(ScalarType::kFloat32, BackendKind::kOpenCL, group));
  EXPECT_FALSE(CompileComputeGraph(ScalarType::kFloat32, BackendKind::kOpenCL,
                                   {{.operation = group, .dependencies = {}}}, &operations));
}

TEST(ParallelExecutionPolicyTest, RejectsMalformedSettingsAndSupportsExplicitSerialComparison) {
  const char *const previous = std::getenv("OMNI_PARALLEL_LINEAR");
  const bool has_previous = previous != nullptr;
  const std::string previous_value = has_previous ? previous : "";
  bool is_enabled = false;
  EXPECT_EQ(unsetenv("OMNI_PARALLEL_LINEAR"), 0);
  EXPECT_TRUE(GetParallelExecutionEnabled(&is_enabled));
  EXPECT_TRUE(is_enabled);
  EXPECT_EQ(setenv("OMNI_PARALLEL_LINEAR", "0", 1), 0);
  EXPECT_TRUE(GetParallelExecutionEnabled(&is_enabled));
  EXPECT_FALSE(is_enabled);
  EXPECT_EQ(setenv("OMNI_PARALLEL_LINEAR", "1", 1), 0);
  EXPECT_TRUE(GetParallelExecutionEnabled(&is_enabled));
  EXPECT_TRUE(is_enabled);
  EXPECT_FALSE(GetParallelExecutionEnabled(nullptr));
  EXPECT_EQ(setenv("OMNI_PARALLEL_LINEAR", "invalid", 1), 0);
  EXPECT_FALSE(GetParallelExecutionEnabled(&is_enabled));
  if (has_previous) {
    EXPECT_EQ(setenv("OMNI_PARALLEL_LINEAR", previous_value.c_str(), 1), 0);
  } else {
    EXPECT_EQ(unsetenv("OMNI_PARALLEL_LINEAR"), 0);
  }
}

} // namespace
} // namespace backend
} // namespace omni_runtime
