#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class ArgmaxBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(ArgmaxBackendTest, test::BackendTypes);

TYPED_TEST(ArgmaxBackendTest, MultirowReductionTailAndTiesReturnFirstIndexExactly) {
  constexpr int32_t kRows = 3;
  constexpr int32_t kColumns = 259;
  auto logits = test::BackendInput<TypeParam>(kRows * kColumns, 4);
  logits[257] = test::BackendValue<TypeParam>(5.0F);
  logits[258] = test::BackendValue<TypeParam>(5.0F);
  logits[kColumns] = test::BackendValue<TypeParam>(4.0F);
  logits[2 * kColumns - 1] = test::BackendValue<TypeParam>(4.0F);
  logits[2 * kColumns + 102] = test::BackendValue<TypeParam>(3.0F);
  const std::vector<int32_t> expected = {257, 0, 102};
  std::array<std::vector<int32_t>, 2U> results;
  for (size_t device = 0U; device < this->devices().size(); ++device) {
    SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
    test::BackendTensor<TypeParam> x;
    test::BackendTensor<int32_t> y;
    ASSERT_TRUE(x.Initialize(this->devices()[device], logits));
    ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<int32_t>(kRows, -23)));
    ASSERT_TRUE(this->devices()[device]->Compute(
        test::BackendType<TypeParam>(),
        backend::Argmax{
            .logits = x.view(), .rows = kRows, .columns = kColumns, .output = y.view()}));
    ASSERT_TRUE(y.Read(&results[device]));
    EXPECT_EQ(results[device], expected);
  }
  EXPECT_EQ(results[0], results[1]);
}

} // namespace
} // namespace omni_runtime::kernels
