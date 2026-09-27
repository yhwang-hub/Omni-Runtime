#include "kernels/backend_test_utils.h"
#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/mha_kernel/mha_opencl_runtime.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class MhaBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(MhaBackendTest, test::BackendTypes);

TYPED_TEST(MhaBackendTest, HeadMajorCausalAndBidirectionalTilesMatchCpu) {
  constexpr int32_t kHeads = 3;
  for (const int32_t tokens : {1, 19}) {
    for (const int32_t dimensions : {7, 37, 256}) {
      const size_t count = kHeads * tokens * dimensions;
      const auto query = test::BackendInput<TypeParam>(count, 1);
      const auto key = test::BackendInput<TypeParam>(count, 5);
      const auto value = test::BackendInput<TypeParam>(count, 13);
      for (const bool causal : {false, true}) {
        SCOPED_TRACE(tokens);
        SCOPED_TRACE(dimensions);
        SCOPED_TRACE(causal);
        std::vector<float> expected;
        ASSERT_TRUE(test::ReferenceAttention(test::BackendFloats(query), test::BackendFloats(key),
                                             test::BackendFloats(value), kHeads, tokens, dimensions,
                                             causal, &expected));
        for (auto &item : expected) {
          item = test::BackendRound<TypeParam>(item);
        }
        std::array<std::vector<float>, 2U> results;
        for (size_t device = 0U; device < this->devices().size(); ++device) {
          SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
          test::BackendTensor<TypeParam> q, k, v, y;
          ASSERT_TRUE(q.Initialize(this->devices()[device], query));
          ASSERT_TRUE(k.Initialize(this->devices()[device], key));
          ASSERT_TRUE(v.Initialize(this->devices()[device], value));
          ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(count)));
          ASSERT_TRUE(this->devices()[device]->Compute(
              test::BackendType<TypeParam>(), backend::MultiHeadAttention{.query = q.view(),
                                                                          .key = k.view(),
                                                                          .value = v.view(),
                                                                          .heads = kHeads,
                                                                          .tokens = tokens,
                                                                          .head_dim = dimensions,
                                                                          .is_causal = causal,
                                                                          .output = y.view()}));
          ASSERT_TRUE(y.ReadFloats(&results[device]));
          test::ExpectBackendClose(results[device], expected,
                                   test::BackendMathTolerance<TypeParam>());
        }
        test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
      }
    }
  }
}

TEST(MhaValidationTest, RejectsOversizedHeadsAndMissingBuffers) {
  backend::OpenCLInvocation call;
  backend::MultiHeadAttention operation{};
  EXPECT_FALSE(runtime::OpenCLOperator<backend::MultiHeadAttention>::Make(operation, nullptr));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::MultiHeadAttention>::Make(operation, &call));
  operation.heads = operation.tokens = 1;
  operation.head_dim = 257;
  EXPECT_FALSE(runtime::OpenCLOperator<backend::MultiHeadAttention>::Make(operation, &call));
  operation.head_dim = 37;
  EXPECT_FALSE(runtime::OpenCLOperator<backend::MultiHeadAttention>::Make(operation, &call));
}

} // namespace
} // namespace omni_runtime::kernels
