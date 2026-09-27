#include "kernels/backend_test_utils.h"
#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/fused_ffn_kernel/fused_ffn_opencl_runtime.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class FusedFfnBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(FusedFfnBackendTest, test::BackendTypes);

TYPED_TEST(FusedFfnBackendTest, TailNeuronsAndRowsKeepUnroundedGeluIntermediate) {
  constexpr int32_t kRows = 3;
  constexpr int32_t kInput = 37;
  constexpr int32_t kIntermediate = 67;
  constexpr int32_t kOutput = 19;
  auto input = test::BackendInput<TypeParam>(kRows * kInput, 1);
  auto first = test::BackendInput<TypeParam>(kIntermediate * kInput, 3);
  const auto first_bias = test::BackendInput<TypeParam>(kIntermediate, 5);
  auto second = test::BackendInput<TypeParam>(kOutput * kIntermediate, 7);
  const auto second_bias = test::BackendInput<TypeParam>(kOutput, 9);
  for (auto *const values : {&input, &first, &second}) {
    for (auto &value : *values) {
      value = test::BackendValue<TypeParam>(test::BackendFloat(value) * 0.25F);
    }
  }
  std::vector<float> expected(kRows * kOutput);
  bool rounded_intermediate_differs = false;
  for (int32_t row = 0; row < kRows; ++row) {
    std::vector<float> intermediate(kIntermediate);
    for (int32_t neuron = 0; neuron < kIntermediate; ++neuron) {
      double sum = test::BackendFloat(first_bias[neuron]);
      for (int32_t d = 0; d < kInput; ++d) {
        sum += double(test::BackendFloat(input[row * kInput + d])) *
               test::BackendFloat(first[neuron * kInput + d]);
      }
      const float value = static_cast<float>(sum);
      intermediate[neuron] = 0.5F * value *
                             (1.0F + std::tanh(0.7978845608028654F *
                                               std::fma(0.044715F * value * value, value, value)));
    }
    for (int32_t neuron = 0; neuron < kOutput; ++neuron) {
      double sum = test::BackendFloat(second_bias[neuron]);
      double rounded_sum = sum;
      for (int32_t d = 0; d < kIntermediate; ++d) {
        const float weight = test::BackendFloat(second[neuron * kIntermediate + d]);
        sum += double(intermediate[d]) * weight;
        rounded_sum += double(test::BackendRound<TypeParam>(intermediate[d])) * weight;
      }
      const float result = test::BackendRound<TypeParam>(static_cast<float>(sum));
      expected[row * kOutput + neuron] = result;
      rounded_intermediate_differs |=
          result != test::BackendRound<TypeParam>(static_cast<float>(rounded_sum));
    }
  }
  if constexpr (!std::is_same_v<TypeParam, float>) {
    EXPECT_TRUE(rounded_intermediate_differs);
  }
  std::array<std::vector<float>, 2U> results;
  for (size_t device = 0U; device < this->devices().size(); ++device) {
    SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
    test::BackendTensor<TypeParam> x, w1, b1, w2, b2, y;
    ASSERT_TRUE(x.Initialize(this->devices()[device], input));
    ASSERT_TRUE(w1.Initialize(this->devices()[device], first));
    ASSERT_TRUE(b1.Initialize(this->devices()[device], first_bias));
    ASSERT_TRUE(w2.Initialize(this->devices()[device], second));
    ASSERT_TRUE(b2.Initialize(this->devices()[device], second_bias));
    ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(expected.size())));
    ASSERT_TRUE(this->devices()[device]->Compute(
        test::BackendType<TypeParam>(), backend::FusedFfn{.input = x.view(),
                                                          .fc1_weight = w1.view(),
                                                          .fc1_bias = b1.view(),
                                                          .fc2_weight = w2.view(),
                                                          .fc2_bias = b2.view(),
                                                          .rows = kRows,
                                                          .input_size = kInput,
                                                          .intermediate_size = kIntermediate,
                                                          .output_size = kOutput,
                                                          .output = y.view()}));
    ASSERT_TRUE(y.ReadFloats(&results[device]));
    test::ExpectBackendClose(results[device], expected, test::BackendMathTolerance<TypeParam>());
  }
  test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
}

TEST(FusedFfnValidationTest, RejectsInvalidLocalMemoryBudgetAndMissingBuffers) {
  backend::OpenCLInvocation call;
  backend::FusedFfn operation{};
  EXPECT_FALSE(runtime::OpenCLOperator<backend::FusedFfn>::Make(operation, nullptr));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::FusedFfn>::Make(operation, &call));
  operation.rows = operation.output_size = 1;
  operation.input_size = operation.intermediate_size = std::numeric_limits<int32_t>::max();
  EXPECT_FALSE(runtime::OpenCLOperator<backend::FusedFfn>::Make(operation, &call));
}

} // namespace
} // namespace omni_runtime::kernels
