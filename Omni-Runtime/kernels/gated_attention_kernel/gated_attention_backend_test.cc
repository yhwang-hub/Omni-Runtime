#include "kernels/backend_test_utils.h"
#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/gated_attention_kernel/gated_attention_opencl_runtime.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class SplitGatedQueryBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(SplitGatedQueryBackendTest, test::BackendTypes);

TYPED_TEST(SplitGatedQueryBackendTest, SplitsEachHeadInsteadOfWholeTokenAtNonzeroOffsets) {
  constexpr int32_t kTokens = 3;
  constexpr int32_t kHeads = 5;
  constexpr int32_t kSize = 37;
  constexpr size_t kCount = kTokens * kHeads * kSize;
  const auto input = test::BackendInput<TypeParam>(2U * kCount, 6);
  std::vector<float> expected_query(kCount), expected_gate(kCount);
  for (size_t index = 0U; index < kCount; ++index) {
    const size_t source = (index / kSize) * (2U * kSize) + index % kSize;
    expected_query[index] = test::BackendFloat(input[source]);
    expected_gate[index] = test::BackendFloat(input[source + kSize]);
  }
  std::array<std::vector<float>, 2U> queries, gates;
  for (size_t device = 0U; device < this->devices().size(); ++device) {
    SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
    test::BackendTensor<TypeParam> x, q, g;
    ASSERT_TRUE(x.Initialize(this->devices()[device], input));
    ASSERT_TRUE(q.Initialize(this->devices()[device], std::vector<TypeParam>(kCount)));
    ASSERT_TRUE(g.Initialize(this->devices()[device], std::vector<TypeParam>(kCount)));
    ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                 backend::SplitGatedQuery{.query_gate = x.view(),
                                                                          .token_count = kTokens,
                                                                          .head_count = kHeads,
                                                                          .head_size = kSize,
                                                                          .query = q.view(),
                                                                          .gate = g.view()}));
    ASSERT_TRUE(q.ReadFloats(&queries[device]));
    ASSERT_TRUE(g.ReadFloats(&gates[device]));
    test::ExpectBackendClose(queries[device], expected_query, {0.0F, 0.0F});
    test::ExpectBackendClose(gates[device], expected_gate, {0.0F, 0.0F});
  }
  test::ExpectBackendClose(queries[0], queries[1], {0.0F, 0.0F});
  test::ExpectBackendClose(gates[0], gates[1], {0.0F, 0.0F});
}

TEST(SplitGatedQueryValidationTest, RejectsOverflowAndMissingBuffers) {
  backend::OpenCLInvocation call;
  backend::SplitGatedQuery operation{};
  EXPECT_FALSE(runtime::OpenCLOperator<backend::SplitGatedQuery>::Make(operation, nullptr));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::SplitGatedQuery>::Make(operation, &call));
  operation.token_count = operation.head_count = operation.head_size =
      std::numeric_limits<int32_t>::max();
  EXPECT_FALSE(runtime::OpenCLOperator<backend::SplitGatedQuery>::Make(operation, &call));
}

} // namespace
} // namespace omni_runtime::kernels
