#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class VisionBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(VisionBackendTest, test::BackendTypes);

TYPED_TEST(VisionBackendTest, PatchPositionRopeAndBiasVariantsMatchBothBackends) {
  constexpr int32_t kGrid = 4;
  constexpr int32_t kTokens = kGrid * kGrid;
  constexpr int32_t kHidden = 80;
  constexpr int32_t kInner = 12;
  const auto patches = test::BackendInput<TypeParam>(kTokens * kInner, 1);
  const auto weight = test::BackendInput<TypeParam>(kHidden * kInner, 3);
  const auto second = test::BackendInput<TypeParam>(weight.size(), 5);
  const auto bias = test::BackendInput<TypeParam>(kHidden, 7);
  const auto position = test::BackendInput<TypeParam>(9U * kHidden, 2);
  const auto qkv = test::BackendInput<TypeParam>(kTokens * kHidden * 3U, 9);
  for (int32_t operation = 0; operation < 7; ++operation) {
    SCOPED_TRACE(operation);
    std::array<std::vector<float>, 2U> outputs;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
      test::BackendTensor<TypeParam> x;
      test::BackendTensor<TypeParam> w;
      test::BackendTensor<TypeParam> w2;
      test::BackendTensor<TypeParam> b;
      test::BackendTensor<TypeParam> p;
      test::BackendTensor<TypeParam> y;
      test::BackendTensor<TypeParam> k;
      test::BackendTensor<TypeParam> v;
      const auto input = operation == 0   ? patches
                         : operation == 2 ? qkv
                                          : test::BackendInput<TypeParam>(kTokens * kHidden, 4);
      ASSERT_TRUE(x.Initialize(this->devices()[device], input));
      ASSERT_TRUE(w.Initialize(this->devices()[device], weight));
      ASSERT_TRUE(w2.Initialize(this->devices()[device], second));
      ASSERT_TRUE(b.Initialize(this->devices()[device], bias));
      ASSERT_TRUE(p.Initialize(this->devices()[device], position));
      ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(kTokens * kHidden)));
      ASSERT_TRUE(k.Initialize(this->devices()[device], std::vector<TypeParam>(kTokens * kHidden)));
      ASSERT_TRUE(v.Initialize(this->devices()[device], std::vector<TypeParam>(kTokens * kHidden)));
      backend::ComputeOperation call;
      switch (operation) {
      case 0:
        call = backend::PairedPatchEmbedding{.input = x.view(),
                                             .first_weight = w.view(),
                                             .second_weight = w2.view(),
                                             .bias = b.view(),
                                             .height = 8,
                                             .width = 8,
                                             .channels = 3,
                                             .output_channels = kHidden,
                                             .spatial_patch = 2,
                                             .output = y.view()};
        break;
      case 1:
        call = backend::GridPositionEmbedding{.input = x.view(),
                                              .position_weight = p.view(),
                                              .grid_height = kGrid,
                                              .grid_width = kGrid,
                                              .position_grid_size = 3,
                                              .hidden_size = kHidden,
                                              .merge_size = 2,
                                              .output = y.view()};
        break;
      case 2:
        call = backend::GridQkvRope{.qkv = x.view(),
                                    .token_count = kTokens,
                                    .grid_height = kGrid,
                                    .grid_width = kGrid,
                                    .merge_size = 2,
                                    .head_count = 2,
                                    .head_size = 40,
                                    .query = y.view(),
                                    .key = k.view(),
                                    .value = v.view()};
        break;
      case 3:
        call = backend::Bias{.input = x.view(),
                             .bias = b.view(),
                             .element_count = input.size(),
                             .row_size = kHidden,
                             .output = x.view()};
        break;
      case 4:
        call = backend::BiasResidual{.input = x.view(),
                                     .bias = b.view(),
                                     .residual = x.view(),
                                     .element_count = input.size(),
                                     .row_size = kHidden,
                                     .output = y.view()};
        break;
      default:
        call = backend::GeluBias{.input = x.view(),
                                 .bias = b.view(),
                                 .element_count = input.size(),
                                 .row_size = kHidden,
                                 .is_approximate = operation == 5,
                                 .output = y.view()};
        break;
      }
      ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(), call));
      ASSERT_TRUE(operation == 3 ? x.ReadFloats(&outputs[device]) : y.ReadFloats(&outputs[device]));
      if (operation == 2) {
        std::vector<float> extra;
        ASSERT_TRUE(k.ReadFloats(&extra));
        outputs[device].insert(outputs[device].end(), extra.begin(), extra.end());
        ASSERT_TRUE(v.ReadFloats(&extra));
        outputs[device].insert(outputs[device].end(), extra.begin(), extra.end());
      }
    }
    test::ExpectBackendClose(outputs[0], outputs[1], test::BackendMathTolerance<TypeParam>());
  }
}

TYPED_TEST(VisionBackendTest, TemporalPatchEmbeddingAndMergePreserveLayout) {
  constexpr int32_t kInner = 24;
  constexpr int32_t kPatches = 8;
  constexpr int32_t kColumns = 7;
  const auto input = test::BackendInput<TypeParam>(kPatches * kInner, 2);
  const auto weight = test::BackendInput<TypeParam>(kColumns * kInner, 3);
  const auto bias = test::BackendInput<TypeParam>(kColumns, 4);
  std::array<std::vector<float>, 2U> outputs;
  for (size_t device = 0U; device < this->devices().size(); ++device) {
    test::BackendTensor<TypeParam> x;
    test::BackendTensor<TypeParam> w;
    test::BackendTensor<TypeParam> b;
    test::BackendTensor<TypeParam> y;
    test::BackendTensor<TypeParam> merged;
    ASSERT_TRUE(x.Initialize(this->devices()[device], input));
    ASSERT_TRUE(w.Initialize(this->devices()[device], weight));
    ASSERT_TRUE(b.Initialize(this->devices()[device], bias));
    ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(kPatches * kColumns)));
    ASSERT_TRUE(
        merged.Initialize(this->devices()[device], std::vector<TypeParam>(kPatches * kColumns)));
    ASSERT_TRUE(this->devices()[device]->Compute(
        test::BackendType<TypeParam>(), backend::PatchEmbedding{.input = x.view(),
                                                                .weight = w.view(),
                                                                .bias = b.view(),
                                                                .frames = 4,
                                                                .height = 4,
                                                                .width = 4,
                                                                .channels = 3,
                                                                .output_channels = kColumns,
                                                                .temporal_patch = 2,
                                                                .spatial_patch = 2,
                                                                .output = y.view()}));
    ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                 backend::PatchMerge{.input = y.view(),
                                                                     .token_count = kPatches,
                                                                     .grid_width = 4,
                                                                     .hidden_size = kColumns,
                                                                     .merge_size = 2,
                                                                     .output = merged.view()}));
    ASSERT_TRUE(merged.ReadFloats(&outputs[device]));
    std::vector<float> original;
    ASSERT_TRUE(y.ReadFloats(&original));
    test::ExpectBackendClose(outputs[device], original, {0.0F, 0.0F});
  }
  test::ExpectBackendClose(outputs[0], outputs[1], test::BackendMathTolerance<TypeParam>());
}

} // namespace
} // namespace omni_runtime::kernels
