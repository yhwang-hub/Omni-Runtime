#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class NormBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(NormBackendTest, test::BackendTypes);

TYPED_TEST(NormBackendTest, RmsNormTailRowsAndInPlaceMatchRoundedCpuReference) {
  constexpr int32_t kRows = 3;
  constexpr float kEpsilon = 1.0e-5F;
  for (const int32_t hidden : {37, 259}) {
    const auto input = test::BackendInput<TypeParam>(kRows * hidden, 2);
    const auto weight = test::BackendInput<TypeParam>(hidden, 7);
    std::vector<float> expected;
    test::BackendRmsReference(input, weight, kRows, hidden, kEpsilon, &expected);
    for (const bool in_place : {false, true}) {
      SCOPED_TRACE(::testing::Message() << hidden << ", in_place=" << in_place);
      std::array<std::vector<float>, 2U> results;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        test::BackendTensor<TypeParam> x;
        test::BackendTensor<TypeParam> w;
        test::BackendTensor<TypeParam> y;
        ASSERT_TRUE(x.Initialize(this->devices()[device], input));
        ASSERT_TRUE(w.Initialize(this->devices()[device], weight));
        ASSERT_TRUE(
            y.Initialize(this->devices()[device],
                         std::vector<TypeParam>(input.size(), test::BackendValue<TypeParam>(-23))));
        auto &output = in_place ? x : y;
        ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                     backend::RmsNorm{.input = x.view(),
                                                                      .weight = w.view(),
                                                                      .rows = kRows,
                                                                      .hidden_size = hidden,
                                                                      .epsilon = kEpsilon,
                                                                      .output = output.view()}));
        ASSERT_TRUE(output.ReadFloats(&results[device]));
        test::ExpectBackendClose(results[device], expected,
                                 test::BackendMathTolerance<TypeParam>());
      }
      test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
    }
  }
}

TYPED_TEST(NormBackendTest, LayerNormWithBiasTailAndInPlaceMatchCpuReference) {
  constexpr int32_t kRows = 3;
  constexpr float kEpsilon = 1.0e-5F;
  for (const int32_t hidden : {37, 259}) {
    const auto input = test::BackendInput<TypeParam>(kRows * hidden, 4);
    const auto weight = test::BackendInput<TypeParam>(hidden, 8);
    const auto bias = test::BackendInput<TypeParam>(hidden, 5);
    std::vector<float> expected(input.size());
    for (int32_t row = 0; row < kRows; ++row) {
      double sum = 0.0;
      double squares = 0.0;
      for (int32_t dimension = 0; dimension < hidden; ++dimension) {
        const float value = test::BackendFloat(input[row * hidden + dimension]);
        sum += value;
        squares += static_cast<double>(value) * value;
      }
      const double mean = sum / hidden;
      const double inverse = 1.0 / std::sqrt(squares / hidden - mean * mean + kEpsilon);
      for (int32_t dimension = 0; dimension < hidden; ++dimension) {
        const size_t index = row * hidden + dimension;
        expected[index] =
            test::BackendRound<TypeParam>((test::BackendFloat(input[index]) - mean) * inverse *
                                              test::BackendFloat(weight[dimension]) +
                                          test::BackendFloat(bias[dimension]));
      }
    }
    for (const bool in_place : {false, true}) {
      std::array<std::vector<float>, 2U> results;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        test::BackendTensor<TypeParam> x;
        test::BackendTensor<TypeParam> w;
        test::BackendTensor<TypeParam> b;
        test::BackendTensor<TypeParam> y;
        ASSERT_TRUE(x.Initialize(this->devices()[device], input));
        ASSERT_TRUE(w.Initialize(this->devices()[device], weight));
        ASSERT_TRUE(b.Initialize(this->devices()[device], bias));
        ASSERT_TRUE(
            y.Initialize(this->devices()[device],
                         std::vector<TypeParam>(input.size(), test::BackendValue<TypeParam>(-23))));
        auto &output = in_place ? x : y;
        ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                     backend::LayerNorm{.input = x.view(),
                                                                        .weight = w.view(),
                                                                        .bias = b.view(),
                                                                        .rows = kRows,
                                                                        .hidden_size = hidden,
                                                                        .epsilon = kEpsilon,
                                                                        .output = output.view()}));
        ASSERT_TRUE(output.ReadFloats(&results[device]));
        test::ExpectBackendClose(results[device], expected,
                                 test::BackendMathTolerance<TypeParam>());
      }
      test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
    }
  }
}

TYPED_TEST(NormBackendTest, FusedResidualRmsNormPreservesRoundedSumAndAliasedOutputs) {
  constexpr int32_t kRows = 3;
  constexpr float kEpsilon = 1.0e-6F;
  for (const int32_t hidden : {37, 259}) {
    const auto input = test::BackendInput<TypeParam>(kRows * hidden, 3);
    const auto residual = test::BackendInput<TypeParam>(input.size(), 6);
    const auto weight = test::BackendInput<TypeParam>(hidden, 9);
    std::vector<TypeParam> sum(input.size());
    for (size_t index = 0U; index < input.size(); ++index) {
      sum[index] = test::BackendValue<TypeParam>(test::BackendFloat(input[index]) +
                                                 test::BackendFloat(residual[index]));
    }
    std::vector<float> expected;
    test::BackendRmsReference(sum, weight, kRows, hidden, kEpsilon, &expected);
    for (const bool in_place : {false, true}) {
      std::array<std::vector<float>, 2U> results;
      std::array<std::vector<float>, 2U> sums;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        test::BackendTensor<TypeParam> x;
        test::BackendTensor<TypeParam> r;
        test::BackendTensor<TypeParam> w;
        test::BackendTensor<TypeParam> s;
        test::BackendTensor<TypeParam> y;
        ASSERT_TRUE(x.Initialize(this->devices()[device], input));
        ASSERT_TRUE(r.Initialize(this->devices()[device], residual));
        ASSERT_TRUE(w.Initialize(this->devices()[device], weight));
        ASSERT_TRUE(
            s.Initialize(this->devices()[device],
                         std::vector<TypeParam>(input.size(), test::BackendValue<TypeParam>(-23))));
        ASSERT_TRUE(
            y.Initialize(this->devices()[device],
                         std::vector<TypeParam>(input.size(), test::BackendValue<TypeParam>(-23))));
        auto &sum_output = in_place ? r : s;
        auto &output = in_place ? x : y;
        ASSERT_TRUE(this->devices()[device]->Compute(
            test::BackendType<TypeParam>(),
            backend::ResidualRmsNorm{.input = x.view(),
                                     .residual = r.view(),
                                     .weight = w.view(),
                                     .rows = kRows,
                                     .hidden_size = hidden,
                                     .epsilon = kEpsilon,
                                     .residual_output = sum_output.view(),
                                     .output = output.view()}));
        ASSERT_TRUE(output.ReadFloats(&results[device]));
        ASSERT_TRUE(sum_output.ReadFloats(&sums[device]));
        test::ExpectBackendClose(sums[device], test::BackendFloats(sum), {0.0F, 0.0F});
        test::ExpectBackendClose(results[device], expected,
                                 test::BackendMathTolerance<TypeParam>());
      }
      test::ExpectBackendClose(sums[0], sums[1], {0.0F, 0.0F});
      test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
    }
  }
}

} // namespace
} // namespace omni_runtime::kernels
