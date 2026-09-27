#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class MatmulBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(MatmulBackendTest, test::BackendTypes);

TYPED_TEST(MatmulBackendTest, GemvAndGemmWithRowColumnInnerTailsMatchCpuProjection) {
  const std::array<std::array<int32_t, 3U>, 3U> shapes = {{{1, 35, 19}, {5, 35, 19}, {3, 64, 65}}};
  for (const auto &shape : shapes) {
    const int32_t rows = shape[0];
    const int32_t inner = shape[1];
    const int32_t columns = shape[2];
    SCOPED_TRACE(::testing::Message() << rows << "x" << inner << "x" << columns);
    const auto input = test::BackendInput<TypeParam>(rows * inner, 1);
    const auto weight = test::BackendInput<TypeParam>(columns * inner, 9);
    std::vector<float> expected(rows * columns);
    for (int32_t row = 0; row < rows; ++row) {
      for (int32_t column = 0; column < columns; ++column) {
        double sum = 0.0;
        for (int32_t dimension = 0; dimension < inner; ++dimension) {
          sum += static_cast<double>(test::BackendFloat(input[row * inner + dimension])) *
                 test::BackendFloat(weight[column * inner + dimension]);
        }
        expected[row * columns + column] = test::BackendRound<TypeParam>(sum);
      }
    }
    for (const bool use_compute : {false, true}) {
      std::array<std::vector<float>, 2U> results;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        test::BackendTensor<TypeParam> x;
        test::BackendTensor<TypeParam> w;
        test::BackendTensor<TypeParam> y;
        ASSERT_TRUE(x.Initialize(this->devices()[device], input));
        ASSERT_TRUE(w.Initialize(this->devices()[device], weight));
        ASSERT_TRUE(y.Initialize(
            this->devices()[device],
            std::vector<TypeParam>(expected.size(), test::BackendValue<TypeParam>(-23))));
        if (use_compute) {
          ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                       backend::Linear{.input = x.view(),
                                                                       .weight = w.view(),
                                                                       .rows = rows,
                                                                       .inner = inner,
                                                                       .columns = columns,
                                                                       .output = y.view()}));
        } else {
          ASSERT_TRUE(this->devices()[device]->LinearMatmul(
              test::BackendType<TypeParam>(), x.view(), w.view(), rows, inner, columns, y.view()));
        }
        ASSERT_TRUE(y.ReadFloats(&results[device]));
        test::ExpectBackendClose(results[device], expected,
                                 test::BackendMathTolerance<TypeParam>());
      }
      test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
    }
  }
}

TYPED_TEST(MatmulBackendTest, ParallelBranchesMatchSerialBitsWithSlicesAndTails) {
  const std::array<std::array<int32_t, 5U>, 6U> shapes = {{{1, 35, 19, 11, 7},
                                                           {1, 64, 65, 9, 17},
                                                           {1, 4096, 64, 32, 16},
                                                           {1, 12288, 16, 16, 8},
                                                           {5, 35, 19, 11, 7},
                                                           {1024, 256, 256, 128, 512}}};
  for (const auto &device : this->devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    for (const auto &shape : shapes) {
      const int32_t rows = shape[0];
      const int32_t inner = shape[1];
      SCOPED_TRACE(::testing::Message() << rows << "x" << inner);
      test::BackendTensor<TypeParam> input;
      ASSERT_TRUE(input.Initialize(device, test::BackendInput<TypeParam>(rows * inner, 1)));
      std::array<test::BackendTensor<TypeParam>, 3U> weights;
      std::array<test::BackendTensor<TypeParam>, 3U> outputs;
      std::array<std::vector<float>, 3U> expected;
      backend::ParallelLinear group;
      for (size_t branch = 0U; branch < 3U; ++branch) {
        const int32_t columns = shape[branch + 2U];
        ASSERT_TRUE(weights[branch].Initialize(
            device,
            test::BackendInput<TypeParam>(columns * inner, static_cast<int32_t>(branch) + 3)));
        ASSERT_TRUE(outputs[branch].Initialize(
            device, std::vector<TypeParam>(rows * columns, test::BackendValue<TypeParam>(-23))));
        const backend::Linear projection = {.input = input.view(),
                                            .weight = weights[branch].view(),
                                            .rows = rows,
                                            .inner = inner,
                                            .columns = columns,
                                            .output = outputs[branch].view()};
        ASSERT_TRUE(device->Compute(test::BackendType<TypeParam>(), projection));
        ASSERT_TRUE(outputs[branch].ReadFloats(&expected[branch]));
        group.projections.push_back(projection);
      }
      for (const size_t count : {2U, 3U}) {
        group.projections.resize(count);
        if (count == 3U) {
          group.projections[2] = {.input = input.view(),
                                  .weight = weights[2].view(),
                                  .rows = rows,
                                  .inner = inner,
                                  .columns = shape[4],
                                  .output = outputs[2].view()};
        }
        for (int32_t repeat = 0; repeat < 2; ++repeat) {
          ASSERT_TRUE(device->Compute(test::BackendType<TypeParam>(), group));
          for (size_t branch = 0U; branch < count; ++branch) {
            std::vector<float> actual;
            ASSERT_TRUE(outputs[branch].ReadFloats(&actual));
            EXPECT_EQ(actual, expected[branch]);
          }
        }
      }
    }
  }
}

TYPED_TEST(MatmulBackendTest, PreparedGraphJoinsBranchesAndReplaysWithoutPreparationSideEffects) {
  for (const auto &device : this->devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<TypeParam> input;
    ASSERT_TRUE(input.Initialize(device, test::BackendInput<TypeParam>(64U, 1)));
    std::array<test::BackendTensor<TypeParam>, 3U> weights;
    std::array<test::BackendTensor<TypeParam>, 3U> outputs;
    std::array<std::vector<float>, 3U> expected;
    std::vector<backend::ComputeNode> nodes;
    for (size_t branch = 0U; branch < 3U; ++branch) {
      ASSERT_TRUE(weights[branch].Initialize(
          device, test::BackendInput<TypeParam>(64U * 17U, static_cast<int32_t>(branch) + 2)));
      ASSERT_TRUE(outputs[branch].Initialize(
          device, std::vector<TypeParam>(17U, test::BackendValue<TypeParam>(-23))));
      const backend::Linear projection = {.input = input.view(),
                                          .weight = weights[branch].view(),
                                          .rows = 1,
                                          .inner = 64,
                                          .columns = 17,
                                          .output = outputs[branch].view()};
      ASSERT_TRUE(device->Compute(test::BackendType<TypeParam>(), projection));
      ASSERT_TRUE(outputs[branch].ReadFloats(&expected[branch]));
      nodes.push_back({.operation = projection, .dependencies = {}});
    }
    test::BackendTensor<TypeParam> result;
    ASSERT_TRUE(
        result.Initialize(device, std::vector<TypeParam>(17U, test::BackendValue<TypeParam>(-23))));
    nodes.push_back({.operation = backend::Add{.lhs = outputs[0].view(),
                                               .rhs = outputs[1].view(),
                                               .element_count = 17U,
                                               .output = result.view()},
                     .dependencies = {0U, 1U, 2U}});
    nodes.push_back({.operation = backend::Add{.lhs = result.view(),
                                               .rhs = outputs[2].view(),
                                               .element_count = 17U,
                                               .output = result.view()},
                     .dependencies = {3U}});
    const auto result_view = result.view();
    const backend::Allocation readback = {.data = result_view.data,
                                          .size_bytes = result_view.size_bytes,
                                          .memory_kind = backend::MemoryKind::kDeviceView,
                                          .offset_bytes = result_view.offset_bytes};
    std::unique_ptr<backend::ExecutionPlan> plan;
    ASSERT_TRUE(
        device->CreateExecutionPlan(test::BackendType<TypeParam>(), nodes, readback, &plan));
    std::vector<float> untouched;
    ASSERT_TRUE(result.ReadFloats(&untouched));
    EXPECT_EQ(untouched, std::vector<float>(17U, -23.0F));
    std::vector<TypeParam> actual(17U);
    for (int32_t iteration = 0; iteration < 3; ++iteration) {
      double elapsed_ms = 0.0;
      ASSERT_TRUE(plan->Run(actual.size() * sizeof(TypeParam), actual.data(), &elapsed_ms));
      EXPECT_GT(elapsed_ms, 0.0);
      for (size_t index = 0U; index < actual.size(); ++index) {
        const float sum = test::BackendRound<TypeParam>(expected[0][index] + expected[1][index]);
        EXPECT_EQ(test::BackendFloat(actual[index]),
                  test::BackendRound<TypeParam>(sum + expected[2][index]));
      }
      ASSERT_TRUE(result.ReadFloats(&untouched));
    }
  }
}

TYPED_TEST(MatmulBackendTest, RejectsInvalidBranchBeforeWritingAnyOutput) {
  for (const auto &device : this->devices()) {
    test::BackendTensor<TypeParam> input;
    test::BackendTensor<TypeParam> weight;
    test::BackendTensor<TypeParam> output;
    ASSERT_TRUE(input.Initialize(device, test::BackendInput<TypeParam>(64U, 1)));
    ASSERT_TRUE(weight.Initialize(device, test::BackendInput<TypeParam>(64U * 17U, 2)));
    ASSERT_TRUE(
        output.Initialize(device, std::vector<TypeParam>(17U, test::BackendValue<TypeParam>(-23))));
    const backend::Linear projection = {.input = input.view(),
                                        .weight = weight.view(),
                                        .rows = 1,
                                        .inner = 64,
                                        .columns = 17,
                                        .output = output.view()};
    backend::ParallelLinear group = {.projections = {projection, projection}};
    EXPECT_FALSE(device->Compute(test::BackendType<TypeParam>(), group));
    std::vector<float> actual;
    ASSERT_TRUE(output.ReadFloats(&actual));
    EXPECT_EQ(actual, std::vector<float>(17U, -23.0F));
    group.projections[1].output = {};
    EXPECT_FALSE(device->Compute(test::BackendType<TypeParam>(), group));
    std::unique_ptr<backend::ExecutionPlan> plan;
    const auto output_view = output.view();
    const backend::Allocation readback = {.data = output_view.data,
                                          .size_bytes = output_view.size_bytes,
                                          .memory_kind = backend::MemoryKind::kDeviceView,
                                          .offset_bytes = output_view.offset_bytes};
    EXPECT_FALSE(device->CreateExecutionPlan(
        test::BackendType<TypeParam>(),
        std::vector<backend::ComputeNode>{{.operation = group, .dependencies = {}}}, readback,
        &plan));
    EXPECT_EQ(plan, nullptr);
    ASSERT_TRUE(output.ReadFloats(&actual));
    EXPECT_EQ(actual, std::vector<float>(17U, -23.0F));
  }
}

TYPED_TEST(MatmulBackendTest, MixedWeightAlignmentKeepsEachBranchesAccumulationOrder) {
  constexpr int32_t kInner = 4096;
  constexpr int32_t kColumns = 17;
  std::vector<TypeParam> inputs(kInner);
  std::vector<TypeParam> weights(kInner * kColumns + 1U);
  for (size_t index = 0U; index < inputs.size(); ++index) {
    inputs[index] = test::BackendValue<TypeParam>(
        static_cast<float>(static_cast<int32_t>(index % 17U) - 8) / 13.0F);
  }
  for (size_t index = 0U; index < weights.size(); ++index) {
    weights[index] = test::BackendValue<TypeParam>(
        static_cast<float>(static_cast<int32_t>((index * 17U) % 79U) - 39) / 37.0F);
  }
  for (const auto &device : this->devices()) {
    const auto input = memory::Buffer::CreateDeviceBufferFromVector(device, inputs);
    const auto weight = memory::Buffer::CreateDeviceBufferFromVector(device, weights);
    ASSERT_NE(input, nullptr);
    ASSERT_NE(weight, nullptr);
    std::array<test::BackendTensor<TypeParam>, 2U> outputs;
    std::array<std::vector<float>, 2U> expected;
    backend::ParallelLinear group;
    for (size_t branch = 0U; branch < 2U; ++branch) {
      ASSERT_TRUE(outputs[branch].Initialize(
          device, std::vector<TypeParam>(kColumns, test::BackendValue<TypeParam>(-23))));
      const backend::Linear projection = {
          .input = input->view(),
          .weight = weight->view().Slice(branch * sizeof(TypeParam),
                                         kInner * kColumns * sizeof(TypeParam)),
          .rows = 1,
          .inner = kInner,
          .columns = kColumns,
          .output = outputs[branch].view()};
      ASSERT_TRUE(device->Compute(test::BackendType<TypeParam>(), projection));
      ASSERT_TRUE(outputs[branch].ReadFloats(&expected[branch]));
      group.projections.push_back(projection);
    }
    ASSERT_TRUE(device->Compute(test::BackendType<TypeParam>(), group));
    for (size_t branch = 0U; branch < 2U; ++branch) {
      std::vector<float> actual;
      ASSERT_TRUE(outputs[branch].ReadFloats(&actual));
      EXPECT_EQ(actual, expected[branch]);
    }
  }
}

} // namespace
} // namespace omni_runtime::kernels
