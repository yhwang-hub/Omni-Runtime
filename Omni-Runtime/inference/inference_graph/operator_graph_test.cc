#include "gtest/gtest.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "Omni-Runtime/inference/inference_graph/operator_graph.h"
#include "Omni-Runtime/memory/buffer.h"

namespace omni_runtime::inference {
namespace {

std::string ConfigPath(const std::string &name) {
  const char *const test_srcdir = std::getenv("TEST_SRCDIR");
  const std::filesystem::path root = test_srcdir == nullptr
                                         ? std::filesystem::path()
                                         : std::filesystem::path(test_srcdir) / "_main";
  return (root / "Omni-Runtime/models/qwen3_vl/graphs" / name).string();
}

class OperatorGraphConfigTest : public ::testing::Test {
protected:
  void SetUp() final {
    path_ = std::filesystem::path(::testing::TempDir()) /
            (std::string("operator_graph_") +
             ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".yaml");
    const backend::DeviceView view{.data = storage_.data(),
                                   .offset_bytes = 0U,
                                   .size_bytes = storage_.size(),
                                   .backend_kind = backend::BackendKind::kOpenCL};
    bindings_.values = {{"positions", view}, {"cache_position", view}};
  }

  void TearDown() final {
    std::error_code error;
    std::filesystem::remove(path_, error);
    EXPECT_FALSE(error) << error.message();
  }

  bool Load(const std::string &source) {
    std::ofstream output(path_);
    output << source;
    OMNI_RETURN_VAL_IF(!output.good(), false);
    output.close();
    return config_.Load(path_.string());
  }

  static constexpr char kStep[] =
      "{op: AdvanceSequence, positions: positions, cache_position: cache_position}";
  std::filesystem::path path_;
  std::array<uint8_t, 128> storage_{};
  OperatorGraphConfig config_;
  OperatorGraphBindings bindings_;
};

TEST_F(OperatorGraphConfigTest, LoadsBothBackendsFromYaml) {
  ASSERT_TRUE(config_.Load(ConfigPath("qwen3-vl-8b-fp16.yaml"))) << config_.error_message();
  EXPECT_EQ(config_.backend(), "cuda");
  const std::string graph_path =
      std::filesystem::absolute(ConfigPath("qwen3-vl-8b-fp16.yaml")).string();
  ASSERT_TRUE(Load("backend: opencl\ngraph: \"" + graph_path + "\"\n")) << config_.error_message();
  EXPECT_EQ(config_.backend(), "opencl");
}

TEST_F(OperatorGraphConfigTest, BindsOperatorsWithoutBackendHandles) {
  ASSERT_TRUE(Load(std::string("backend: opencl\nsteps:\n  - ") + kStep));
  std::vector<backend::ComputeNode> nodes;
  ASSERT_TRUE(config_.Append("steps", bindings_, {}, &nodes)) << config_.error_message();
  ASSERT_EQ(nodes.size(), 1U);
  EXPECT_TRUE(nodes.front().dependencies.empty());
  ASSERT_TRUE(std::holds_alternative<backend::AdvanceSequence>(nodes.front().operation));
  const auto &operation = std::get<backend::AdvanceSequence>(nodes.front().operation);
  EXPECT_EQ(operation.positions.data, storage_.data());
  EXPECT_EQ(operation.positions.backend_kind, backend::BackendKind::kOpenCL);
}

TEST_F(OperatorGraphConfigTest, ExpandsRepeatConditionAndAliases) {
  ASSERT_TRUE(Load(std::string("backend: cuda\nsteps:\n"
                               "  - repeat: count\n    as: layer\n    body:\n"
                               "      - &step ") +
                   kStep +
                   "\n      - {op: AdvanceSequence, when: enabled, positions: positions, "
                   "cache_position: cache_position}\n  - *step\n"));
  bindings_.values["count"] = int32_t{2};
  bindings_.values["enabled"] = false;
  std::vector<int32_t> visited;
  const OperatorIterationBinder bind = [&visited](const std::string &variable, const int32_t index,
                                                  OperatorGraphBindings *const bindings) {
    OMNI_RETURN_VAL_IF(bindings == nullptr || variable != "layer", false);
    visited.push_back(index);
    return true;
  };
  std::vector<backend::ComputeNode> nodes;
  ASSERT_TRUE(config_.Append("steps", bindings_, bind, &nodes));
  EXPECT_EQ(visited, (std::vector<int32_t>{0, 1}));
  ASSERT_EQ(nodes.size(), 3U);
  EXPECT_EQ(nodes[1].dependencies, (std::vector<uint32_t>{0U}));
  EXPECT_EQ(nodes[2].dependencies, (std::vector<uint32_t>{1U}));
}

TEST_F(OperatorGraphConfigTest, JoinsAllParallelBranchesAcrossSections) {
  ASSERT_TRUE(Load(std::string("backend: cuda\nfork:\n  - parallel:\n      - ") + kStep +
                   "\n      - " + kStep + "\njoin:\n  - " + kStep));
  std::vector<backend::ComputeNode> nodes;
  ASSERT_TRUE(config_.Append("fork", bindings_, {}, &nodes));
  ASSERT_TRUE(config_.Append("join", bindings_, {}, &nodes));
  ASSERT_EQ(nodes.size(), 3U);
  EXPECT_TRUE(nodes[0].dependencies.empty());
  EXPECT_TRUE(nodes[1].dependencies.empty());
  EXPECT_EQ(nodes[2].dependencies, (std::vector<uint32_t>{0U, 1U}));
}

TEST_F(OperatorGraphConfigTest, ResolvesIndexedWeightsAndChecksExactScalarTypes) {
  ASSERT_TRUE(Load("backend: cuda\nsteps:\n"
                   "  - repeat: 2\n    as: layer\n    body:\n"
                   "      - {op: Linear, input: positions, weight: 'blk.{layer}.weight', "
                   "rows: rows, inner: 2, columns: 2, output: positions}\n"));
  bindings_.values["rows"] = int32_t{1};
  std::vector<std::string> names;
  bindings_.resolve_view = [this, &names](const std::string &name,
                                          backend::DeviceView *const view) {
    OMNI_RETURN_VAL_IF(view == nullptr, false);
    names.push_back(name);
    *view = std::get<backend::DeviceView>(bindings_.values.at("positions"));
    return true;
  };
  std::vector<backend::ComputeNode> nodes;
  ASSERT_TRUE(config_.Append("steps", bindings_, {}, &nodes));
  EXPECT_EQ(names, (std::vector<std::string>{"blk.0.weight", "blk.1.weight"}));
  bindings_.values["rows"] = true;
  EXPECT_FALSE(config_.Append("steps", bindings_, {}, &nodes));
  EXPECT_EQ(nodes.size(), 2U);
  EXPECT_FALSE(config_.error_message().empty());
}

TEST_F(OperatorGraphConfigTest, KeepsLegacyDeltaNetFlagsOptional) {
  const std::string source =
      "backend: cuda\nsteps:\n  - {op: GatedDeltaNet, qkv: positions, alpha: positions, "
      "beta: positions, decay_weight: positions, time_bias: positions, state: positions, "
      "token_count: 4, key_head_count: 1, value_head_count: 2, head_size: 128, "
      "epsilon: 0.000001, output: positions";
  for (const bool is_reference : {false, true}) {
    ASSERT_TRUE(Load(source +
                     (is_reference ? ", is_chunked: true, is_log_decay: true}\n" : "}\n")))
        << config_.error_message();
    std::vector<backend::ComputeNode> nodes;
    ASSERT_TRUE(config_.Append("steps", bindings_, {}, &nodes)) << config_.error_message();
    ASSERT_EQ(nodes.size(), 1U);
    const auto &operation = std::get<backend::GatedDeltaNet>(nodes.front().operation);
    EXPECT_EQ(operation.is_chunked, is_reference);
    EXPECT_EQ(operation.is_log_decay, is_reference);
  }
}

TEST_F(OperatorGraphConfigTest, PreservesFloatBiasDefaultAndBindsActivationBias) {
  ASSERT_TRUE(Load(
      "backend: cuda\nsteps:\n"
      "  - {op: Linear, input: positions, weight: positions, bias: positions, "
      "rows: 3, inner: 8, columns: 16, output: positions}\n"
      "  - {op: Linear, input: positions, weight: positions, bias: positions, "
      "is_float_bias: false, rows: 3, inner: 8, columns: 16, output: positions}\n"));
  std::vector<backend::ComputeNode> nodes;
  ASSERT_TRUE(config_.Append("steps", bindings_, {}, &nodes)) << config_.error_message();
  ASSERT_EQ(nodes.size(), 2U);
  EXPECT_TRUE(std::get<backend::Linear>(nodes[0].operation).is_float_bias);
  EXPECT_FALSE(std::get<backend::Linear>(nodes[1].operation).is_float_bias);
}

TEST_F(OperatorGraphConfigTest, BindsOptionalFp8BlockReductionGroups) {
  const std::string source =
      "backend: cuda\nsteps:\n  - {op: Fp8Linear, input: positions, weight: positions, "
      "weight_scales: positions, rows: 4, inner: 6144, columns: 5120, output: positions";
  bindings_.values["groups"] = int32_t{3};
  for (const bool has_groups : {false, true}) {
    ASSERT_TRUE(Load(source + (has_groups ? ", block_reduction_groups: groups}\n" : "}\n")))
        << config_.error_message();
    std::vector<backend::ComputeNode> nodes;
    ASSERT_TRUE(config_.Append("steps", bindings_, {}, &nodes)) << config_.error_message();
    ASSERT_EQ(nodes.size(), 1U);
    EXPECT_EQ(std::get<backend::Fp8Linear>(nodes.front().operation).block_reduction_groups,
              has_groups ? 3 : 1);
  }
}

TEST_F(OperatorGraphConfigTest, RejectsInvalidSchemasAndRecursiveAliases) {
  for (const std::string source :
       {"backend: cpu\nsteps: []", "backend: cuda\nsteps: []", "backend: cuda\nsteps: [",
        "backend: cuda\nsteps: [{op: Missing}]",
        "backend: cuda\nsteps: [{op: Linear, input: hidden}]",
        "backend: cuda\nsteps: [{op: AdvanceSequence, positions: x, cache_position: y, typo: 1}]",
        "backend: cuda\nsteps: [{op: AdvanceSequence, positions: x, positions: y, cache_position: "
        "y}]",
        "backend: cuda\nsteps: &loop [*loop]",
        "backend: cuda\nsteps: [{repeat: 2, as: layer, body: []}]",
        "backend: cuda\nsteps: [{parallel: [], op: Linear}]"}) {
    EXPECT_FALSE(Load(source)) << source;
    EXPECT_FALSE(config_.error_message().empty()) << source;
  }
}

TEST_F(OperatorGraphConfigTest, RejectsMissingBindingsAndInvalidLoopCountsAtomically) {
  ASSERT_TRUE(Load(std::string("backend: cuda\nsteps:\n  - repeat: count\n    as: layer\n"
                               "    body:\n      - ") +
                   kStep));
  std::vector<backend::ComputeNode> nodes;
  bindings_.values["count"] = int32_t{-1};
  EXPECT_FALSE(config_.Append("steps", bindings_, {}, &nodes));
  bindings_.values["count"] = int32_t{4097};
  EXPECT_FALSE(config_.Append("steps", bindings_, {}, &nodes));
  bindings_.values["count"] = int32_t{1};
  bindings_.values.erase("positions");
  EXPECT_FALSE(config_.Append("steps", bindings_, {}, &nodes));
  EXPECT_FALSE(config_.Append("missing", bindings_, {}, &nodes));
  EXPECT_FALSE(config_.Append("steps", bindings_, {}, nullptr));
  EXPECT_TRUE(nodes.empty());
}

TEST_F(OperatorGraphConfigTest, RejectsUseAfterFailedLoadAndInvalidNumericValues) {
  std::vector<backend::ComputeNode> nodes;
  EXPECT_FALSE(config_.Append("steps", bindings_, {}, &nodes));
  ASSERT_TRUE(Load(std::string("backend: cuda\nsteps:\n  - ") + kStep));
  EXPECT_FALSE(Load("backend: cuda\nsteps: [{op: Invalid}]"));
  EXPECT_FALSE(config_.Append("steps", bindings_, {}, &nodes));
  for (const std::string count : {"-1", "18446744073709551616", "missing"}) {
    ASSERT_TRUE(Load(
        "backend: cuda\nsteps:\n  - {op: Add, lhs: positions, rhs: positions, output: positions, "
        "element_count: " +
        count + "}"));
    EXPECT_FALSE(config_.Append("steps", bindings_, {}, &nodes));
    EXPECT_TRUE(nodes.empty());
  }
  ASSERT_TRUE(Load("backend: cuda\nsteps:\n  - {op: RmsNorm, input: positions, weight: positions, "
                   "output: positions, rows: 1, hidden_size: 2, epsilon: .nan}"));
  EXPECT_FALSE(config_.Append("steps", bindings_, {}, &nodes));
}

TEST_F(OperatorGraphConfigTest, ExecutesConditionalBlocksAndRejectsExcessiveExpansion) {
  ASSERT_TRUE(
      Load(std::string("backend: cuda\nsteps:\n  - when: enabled\n    body:\n      - ") + kStep));
  bindings_.values["enabled"] = false;
  std::vector<backend::ComputeNode> nodes;
  ASSERT_TRUE(config_.Append("steps", bindings_, {}, &nodes));
  EXPECT_TRUE(nodes.empty());
  bindings_.values["enabled"] = true;
  ASSERT_TRUE(config_.Append("steps", bindings_, {}, &nodes));
  EXPECT_EQ(nodes.size(), 1U);
  ASSERT_TRUE(Load(std::string("backend: cuda\nsteps:\n"
                               "  - repeat: 129\n    as: outer\n    body:\n"
                               "      - repeat: 128\n        as: inner\n        body:\n"
                               "          - ") +
                   kStep));
  EXPECT_FALSE(config_.Append("steps", bindings_, {}, &nodes));
  EXPECT_EQ(nodes.size(), 1U);
}

TEST_F(OperatorGraphConfigTest, ExecutesAndReusesYamlOperatorNodesOnBothBackends) {
  for (const std::string name : {"cuda", "opencl"}) {
    SCOPED_TRACE(name);
    const auto device = backend::CreateComputeBackend(name);
    if (device == nullptr) {
      GTEST_SKIP() << name << " is unavailable";
    }
    ASSERT_TRUE(
        Load("backend: " + name +
             "\nsteps:\n  - repeat: 2\n    as: step\n    body:\n"
             "      - {op: Add, lhs: state, rhs: increment, element_count: 4, output: state}"));
    const std::array<float, 4> input{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> increment{0.5F, 1.0F, 1.5F, 2.0F};
    const auto state =
        memory::Buffer::CreateOwnedBuffer(device, sizeof(input), backend::MemoryKind::kDevice);
    const auto delta =
        memory::Buffer::CreateOwnedBuffer(device, sizeof(increment), backend::MemoryKind::kDevice);
    ASSERT_NE(state, nullptr);
    ASSERT_NE(delta, nullptr);
    ASSERT_TRUE(state->CopyFromHost(input.data(), sizeof(input)));
    ASSERT_TRUE(delta->CopyFromHost(increment.data(), sizeof(increment)));
    bindings_.values["state"] = state->view();
    bindings_.values["increment"] = delta->view();
    std::vector<backend::ComputeNode> nodes;
    ASSERT_TRUE(config_.Append("steps", bindings_, {}, &nodes));
    ASSERT_EQ(nodes.size(), 2U);
    OperatorGraph graph;
    ASSERT_TRUE(graph.Prepare("add", device, backend::ScalarType::kFloat32, nodes))
        << graph.error_message();
    std::array<float, 4> output{};
    for (int32_t run = 1; run <= 2; ++run) {
      ASSERT_TRUE(graph.Run()) << graph.error_message();
      ASSERT_TRUE(state->CopyToHost(sizeof(output), output.data()));
      for (size_t index = 0U; index < output.size(); ++index) {
        EXPECT_FLOAT_EQ(output[index], input[index] + 2 * run * increment[index]);
      }
    }
    EXPECT_FALSE(graph.Prepare("invalid", nullptr, backend::ScalarType::kFloat32, nodes));
    EXPECT_FALSE(graph.error_message().empty());
    EXPECT_FALSE(graph.Run());
  }
}

TEST(OperatorGraphTest, RejectsUnpreparedOperatorGraph) {
  OperatorGraph graph;
  EXPECT_FALSE(graph.Run());
  EXPECT_FALSE(graph.error_message().empty());
}

} // namespace
} // namespace omni_runtime::inference
