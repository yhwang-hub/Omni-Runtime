#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class AddBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(AddBackendTest, test::BackendTypes);

TYPED_TEST(AddBackendTest, TailSlicesAndAliasedOutputsMatchExactCpuSum) {
  constexpr size_t kCount = 517U;
  const auto lhs = test::BackendInput<TypeParam>(kCount, 1);
  const auto rhs = test::BackendInput<TypeParam>(kCount, 7);
  std::vector<float> expected(kCount);
  for (size_t index = 0U; index < kCount; ++index) {
    expected[index] = test::BackendRound<TypeParam>(test::BackendFloat(lhs[index]) +
                                                    test::BackendFloat(rhs[index]));
  }
  for (const int32_t alias : {0, 1, 2}) {
    SCOPED_TRACE(alias);
    std::array<std::vector<float>, 2U> results;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
      test::BackendTensor<TypeParam> x;
      test::BackendTensor<TypeParam> r;
      test::BackendTensor<TypeParam> y;
      ASSERT_TRUE(x.Initialize(this->devices()[device], lhs));
      ASSERT_TRUE(r.Initialize(this->devices()[device], rhs));
      ASSERT_TRUE(y.Initialize(this->devices()[device],
                               std::vector<TypeParam>(kCount, test::BackendValue<TypeParam>(-23))));
      auto &output = alias == 1 ? x : alias == 2 ? r : y;
      ASSERT_TRUE(this->devices()[device]->Compute(
          test::BackendType<TypeParam>(),
          backend::Add{
              .lhs = x.view(), .rhs = r.view(), .element_count = kCount, .output = output.view()}));
      ASSERT_TRUE(output.ReadFloats(&results[device]));
      test::ExpectBackendClose(results[device], expected, {0.0F, 0.0F});
    }
    test::ExpectBackendClose(results[0], results[1], {0.0F, 0.0F});
  }
}

} // namespace
} // namespace omni_runtime::kernels
