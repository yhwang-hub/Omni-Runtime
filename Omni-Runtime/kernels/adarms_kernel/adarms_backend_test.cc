#include "kernels/backend_test_utils.h"
#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/adarms_kernel/adarms_opencl_runtime.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class AdaRmsNormBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(AdaRmsNormBackendTest, test::BackendTypes);

TYPED_TEST(AdaRmsNormBackendTest, TailRowsBroadcastAndInPlaceKeepFp32Modulation) {
  constexpr int32_t kRows = 3;
  constexpr float kEpsilon = 1.0e-5F;
  for (const int32_t hidden : {37, 513}) {
    const auto input = test::BackendInput<TypeParam>(kRows * hidden, 3);
    const auto scale = test::BackendInput<TypeParam>(hidden, 5);
    const auto shift = test::BackendInput<TypeParam>(hidden, 9);
    std::vector<float> expected(input.size());
    for (int32_t row = 0; row < kRows; ++row) {
      double sum = 0.0;
      for (int32_t d = 0; d < hidden; ++d) {
        const float value = test::BackendFloat(input[row * hidden + d]);
        sum += double(value) * value;
      }
      const float inverse = 1.0F / std::sqrt(static_cast<float>(sum / hidden) + kEpsilon);
      for (int32_t d = 0; d < hidden; ++d) {
        const size_t index = row * hidden + d;
        expected[index] = test::BackendRound<TypeParam>(
            std::fma(test::BackendFloat(input[index]) * inverse,
                     1.0F + test::BackendFloat(scale[d]), test::BackendFloat(shift[d])));
      }
    }
    for (const bool alias : {false, true}) {
      std::array<std::vector<float>, 2U> results;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        SCOPED_TRACE(hidden);
        SCOPED_TRACE(alias);
        test::BackendTensor<TypeParam> x, s, b, y;
        ASSERT_TRUE(x.Initialize(this->devices()[device], input));
        ASSERT_TRUE(s.Initialize(this->devices()[device], scale));
        ASSERT_TRUE(b.Initialize(this->devices()[device], shift));
        ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(input.size())));
        auto &output = alias ? x : y;
        ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                     backend::AdaRmsNorm{.input = x.view(),
                                                                         .scale = s.view(),
                                                                         .shift = b.view(),
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

TEST(AdaRmsNormValidationTest, RejectsInvalidDimensionsAndEpsilon) {
  backend::OpenCLInvocation call;
  backend::AdaRmsNorm operation{};
  EXPECT_FALSE(runtime::OpenCLOperator<backend::AdaRmsNorm>::Make(operation, &call));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::AdaRmsNorm>::Make(operation, nullptr));
  operation.rows = 1;
  operation.hidden_size = 1;
  operation.epsilon = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(runtime::OpenCLOperator<backend::AdaRmsNorm>::Make(operation, &call));
}

} // namespace
} // namespace omni_runtime::kernels
