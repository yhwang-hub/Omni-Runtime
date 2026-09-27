#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class EmbeddingBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(EmbeddingBackendTest, test::BackendTypes);

TYPED_TEST(EmbeddingBackendTest, RepeatedUnsortedIndicesAndHiddenTailMatchCpuGather) {
  for (const int32_t kHidden : {37, 4096}) {
    const std::vector<int32_t> indices = {6, 0, 3, 6, 1};
    const auto weight = test::BackendInput<TypeParam>(7U * kHidden, 3);
    std::vector<float> expected(indices.size() * kHidden);
    for (size_t token = 0U; token < indices.size(); ++token) {
      for (int32_t dimension = 0; dimension < kHidden; ++dimension) {
        expected[token * kHidden + dimension] =
            test::BackendFloat(weight[indices[token] * kHidden + dimension]);
      }
    }
    std::array<std::vector<float>, 2U> results;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
      test::BackendTensor<int32_t> ids;
      test::BackendTensor<TypeParam> w;
      test::BackendTensor<TypeParam> y;
      ASSERT_TRUE(ids.Initialize(this->devices()[device], indices));
      ASSERT_TRUE(w.Initialize(this->devices()[device], weight));
      ASSERT_TRUE(y.Initialize(
          this->devices()[device],
          std::vector<TypeParam>(expected.size(), test::BackendValue<TypeParam>(-23))));
      ASSERT_TRUE(this->devices()[device]->Compute(
          test::BackendType<TypeParam>(),
          backend::Embedding{.indices = ids.view(),
                             .weight = w.view(),
                             .token_count = static_cast<int32_t>(indices.size()),
                             .hidden_size = kHidden,
                             .output = y.view()}));
      ASSERT_TRUE(y.ReadFloats(&results[device]));
      test::ExpectBackendClose(results[device], expected, {0.0F, 0.0F});
    }
    test::ExpectBackendClose(results[0], results[1], {0.0F, 0.0F});
  }
}

} // namespace
} // namespace omni_runtime::kernels
