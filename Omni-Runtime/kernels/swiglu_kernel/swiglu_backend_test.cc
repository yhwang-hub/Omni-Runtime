#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class SwiGLUBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(SwiGLUBackendTest, test::BackendTypes);

TYPED_TEST(SwiGLUBackendTest, TailSlicesAndAliasedGateMatchCpuActivation) {
  constexpr size_t kCount = 515U;
  const auto gate = test::BackendInput<TypeParam>(kCount, 2);
  const auto up = test::BackendInput<TypeParam>(kCount, 5);
  std::vector<float> expected(kCount);
  for (size_t index = 0U; index < kCount; ++index) {
    const float value = test::BackendFloat(gate[index]);
    const float silu = test::BackendRound<TypeParam>(value / (1.0F + std::exp(-value)));
    expected[index] = test::BackendRound<TypeParam>(silu * test::BackendFloat(up[index]));
  }
  for (const bool in_place : {false, true}) {
    std::array<std::vector<float>, 2U> results;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
      test::BackendTensor<TypeParam> g;
      test::BackendTensor<TypeParam> u;
      test::BackendTensor<TypeParam> y;
      ASSERT_TRUE(g.Initialize(this->devices()[device], gate));
      ASSERT_TRUE(u.Initialize(this->devices()[device], up));
      ASSERT_TRUE(y.Initialize(this->devices()[device],
                               std::vector<TypeParam>(kCount, test::BackendValue<TypeParam>(-23))));
      auto &output = in_place ? g : y;
      ASSERT_TRUE(this->devices()[device]->Compute(
          test::BackendType<TypeParam>(),
          backend::SwiGLU{
              .gate = g.view(), .up = u.view(), .element_count = kCount, .output = output.view()}));
      ASSERT_TRUE(output.ReadFloats(&results[device]));
      test::ExpectBackendClose(results[device], expected, test::BackendMathTolerance<TypeParam>());
    }
    test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
  }
}

} // namespace
} // namespace omni_runtime::kernels
