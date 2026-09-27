#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels {
namespace {

template <typename T>
void RopeReference(const std::vector<T> &input, const std::vector<int32_t> &positions,
                   const int32_t tokens, const int32_t heads, const int32_t head_dim,
                   const int32_t height_section, const int32_t width_section,
                   const float frequency_base, std::vector<float> *const output) {
  output->resize(input.size());
  const int32_t half = head_dim / 2;
  for (size_t index = 0U; index < input.size(); ++index) {
    const int32_t dimension = index % head_dim;
    const int32_t pair = dimension % half;
    const int32_t axis = pair % 3 == 1 && pair < 3 * height_section  ? 1
                         : pair % 3 == 2 && pair < 3 * width_section ? 2
                                                                     : 0;
    const int32_t token = index / (heads * head_dim);
    const float angle =
        positions[axis * tokens + token] * std::pow(frequency_base, -2.0F * pair / head_dim);
    const size_t partner = dimension < half ? index + half : index - half;
    const float cosine = test::BackendRound<T>(std::cos(angle));
    const float sine = test::BackendRound<T>(std::sin(angle));
    const float first = test::BackendRound<T>(test::BackendFloat(input[index]) * cosine);
    const float second = test::BackendRound<T>((dimension < half ? -1.0F : 1.0F) *
                                               test::BackendFloat(input[partner]) * sine);
    (*output)[index] = test::BackendRound<T>(first + second);
  }
}

template <typename T> class RopeBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(RopeBackendTest, test::BackendTypes);

TYPED_TEST(RopeBackendTest, MultiAxisTokenHeadTailsMatchCpuRotationAndRounding) {
  constexpr int32_t kTokens = 3;
  constexpr int32_t kQueryHeads = 4;
  constexpr int32_t kKvHeads = 2;
  constexpr float kBase = 5000000.0F;
  const std::vector<int32_t> positions = {0, 7, 47, 13, 1, 29, 2, 31, 5};
  for (const int32_t head_dim : {72, 128}) {
    const int32_t height_section = head_dim == 72 ? 11 : 20;
    const int32_t width_section = head_dim == 72 ? 8 : 12;
    const int32_t temporal_section = head_dim / 2 - height_section - width_section;
    const auto query = test::BackendInput<TypeParam>(kTokens * kQueryHeads * head_dim, 3);
    const auto key = test::BackendInput<TypeParam>(kTokens * kKvHeads * head_dim, 8);
    std::vector<float> expected_query;
    std::vector<float> expected_key;
    RopeReference(query, positions, kTokens, kQueryHeads, head_dim, height_section, width_section,
                  kBase, &expected_query);
    RopeReference(key, positions, kTokens, kKvHeads, head_dim, height_section, width_section, kBase,
                  &expected_key);
    std::array<std::vector<float>, 2U> queries;
    std::array<std::vector<float>, 2U> keys;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
      test::BackendTensor<TypeParam> q;
      test::BackendTensor<TypeParam> k;
      test::BackendTensor<int32_t> p;
      test::BackendTensor<TypeParam> qr;
      test::BackendTensor<TypeParam> kr;
      ASSERT_TRUE(q.Initialize(this->devices()[device], query));
      ASSERT_TRUE(k.Initialize(this->devices()[device], key));
      ASSERT_TRUE(p.Initialize(this->devices()[device], positions));
      ASSERT_TRUE(
          qr.Initialize(this->devices()[device],
                        std::vector<TypeParam>(query.size(), test::BackendValue<TypeParam>(-23))));
      ASSERT_TRUE(
          kr.Initialize(this->devices()[device],
                        std::vector<TypeParam>(key.size(), test::BackendValue<TypeParam>(-23))));
      ASSERT_TRUE(this->devices()[device]->Compute(
          test::BackendType<TypeParam>(),
          backend::MultiAxisRope{.query = q.view(),
                                 .key = k.view(),
                                 .position_ids = p.view(),
                                 .tokens = kTokens,
                                 .query_heads = kQueryHeads,
                                 .key_value_heads = kKvHeads,
                                 .head_dim = head_dim,
                                 .temporal_section = temporal_section,
                                 .height_section = height_section,
                                 .width_section = width_section,
                                 .frequency_base = kBase,
                                 .query_output = qr.view(),
                                 .key_output = kr.view()}));
      ASSERT_TRUE(qr.ReadFloats(&queries[device]));
      ASSERT_TRUE(kr.ReadFloats(&keys[device]));
      test::ExpectBackendClose(queries[device], expected_query,
                               test::BackendMathTolerance<TypeParam>());
      test::ExpectBackendClose(keys[device], expected_key, test::BackendMathTolerance<TypeParam>());
    }
    test::ExpectBackendClose(queries[0], queries[1], test::BackendMathTolerance<TypeParam>());
    test::ExpectBackendClose(keys[0], keys[1], test::BackendMathTolerance<TypeParam>());
  }
}

TYPED_TEST(RopeBackendTest, NormalizeRopeCacheUsesResidentPositionAndAliasedQuery) {
  constexpr int32_t kQueryHeads = 4;
  constexpr int32_t kKvHeads = 2;
  constexpr int32_t kHead = 72;
  constexpr int32_t kCapacity = 5;
  constexpr int32_t kPosition = 2;
  constexpr int32_t kHeightSection = 11;
  constexpr int32_t kWidthSection = 8;
  constexpr float kBase = 5000000.0F;
  constexpr float kEpsilon = 1.0e-6F;
  const auto query = test::BackendInput<TypeParam>(kQueryHeads * kHead, 2);
  const auto key = test::BackendInput<TypeParam>(kKvHeads * kHead, 6);
  const auto value = test::BackendInput<TypeParam>(key.size(), 9);
  const auto query_weight = test::BackendInput<TypeParam>(kHead, 1);
  const auto key_weight = test::BackendInput<TypeParam>(kHead, 7);
  const std::vector<int32_t> positions = {47, 13, 29};
  std::vector<float> normalized_query;
  std::vector<float> normalized_key;
  test::BackendRmsReference(query, query_weight, kQueryHeads, kHead, kEpsilon, &normalized_query);
  test::BackendRmsReference(key, key_weight, kKvHeads, kHead, kEpsilon, &normalized_key);
  std::vector<TypeParam> qn(query.size());
  std::vector<TypeParam> kn(key.size());
  for (size_t index = 0U; index < qn.size(); ++index) {
    qn[index] = test::BackendValue<TypeParam>(normalized_query[index]);
  }
  for (size_t index = 0U; index < kn.size(); ++index) {
    kn[index] = test::BackendValue<TypeParam>(normalized_key[index]);
  }
  std::vector<float> expected_query;
  std::vector<float> rotated_key;
  RopeReference(qn, positions, 1, kQueryHeads, kHead, kHeightSection, kWidthSection, kBase,
                &expected_query);
  RopeReference(kn, positions, 1, kKvHeads, kHead, kHeightSection, kWidthSection, kBase,
                &rotated_key);
  const std::vector<TypeParam> initial(kCapacity * key.size(), test::BackendValue<TypeParam>(-3));
  std::vector<float> expected_key = test::BackendFloats(initial);
  std::vector<float> expected_value = expected_key;
  for (size_t index = 0U; index < key.size(); ++index) {
    expected_key[kPosition * key.size() + index] = rotated_key[index];
    expected_value[kPosition * key.size() + index] = test::BackendFloat(value[index]);
  }
  for (const bool in_place : {false, true}) {
    std::array<std::vector<float>, 2U> queries;
    std::array<std::vector<float>, 2U> keys;
    std::array<std::vector<float>, 2U> values;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
      test::BackendTensor<TypeParam> q;
      test::BackendTensor<TypeParam> k;
      test::BackendTensor<TypeParam> v;
      test::BackendTensor<TypeParam> qw;
      test::BackendTensor<TypeParam> kw;
      test::BackendTensor<int32_t> state;
      test::BackendTensor<TypeParam> qr;
      test::BackendTensor<TypeParam> kc;
      test::BackendTensor<TypeParam> vc;
      ASSERT_TRUE(q.Initialize(this->devices()[device], query));
      ASSERT_TRUE(k.Initialize(this->devices()[device], key));
      ASSERT_TRUE(v.Initialize(this->devices()[device], value));
      ASSERT_TRUE(qw.Initialize(this->devices()[device], query_weight));
      ASSERT_TRUE(kw.Initialize(this->devices()[device], key_weight));
      ASSERT_TRUE(state.Initialize(this->devices()[device], {47, 13, 29, kPosition}));
      ASSERT_TRUE(
          qr.Initialize(this->devices()[device],
                        std::vector<TypeParam>(query.size(), test::BackendValue<TypeParam>(-23))));
      ASSERT_TRUE(kc.Initialize(this->devices()[device], initial));
      ASSERT_TRUE(vc.Initialize(this->devices()[device], initial));
      auto &output = in_place ? q : qr;
      ASSERT_TRUE(this->devices()[device]->Compute(
          test::BackendType<TypeParam>(),
          backend::NormalizeRopeCache{
              .query = q.view(),
              .key = k.view(),
              .value = v.view(),
              .query_weight = qw.view(),
              .key_weight = kw.view(),
              .positions = state.view().Slice(0U, 3U * sizeof(int32_t)),
              .cache_position = state.view().Slice(3U * sizeof(int32_t), sizeof(int32_t)),
              .query_heads = kQueryHeads,
              .key_value_heads = kKvHeads,
              .head_dim = kHead,
              .cache_capacity = kCapacity,
              .temporal_section = kHead / 2 - kHeightSection - kWidthSection,
              .height_section = kHeightSection,
              .width_section = kWidthSection,
              .frequency_base = kBase,
              .epsilon = kEpsilon,
              .query_output = output.view(),
              .key_cache = kc.view(),
              .value_cache = vc.view()}));
      ASSERT_TRUE(output.ReadFloats(&queries[device]));
      ASSERT_TRUE(kc.ReadFloats(&keys[device]));
      ASSERT_TRUE(vc.ReadFloats(&values[device]));
      test::ExpectBackendClose(queries[device], expected_query,
                               test::BackendMathTolerance<TypeParam>());
      test::ExpectBackendClose(keys[device], expected_key, test::BackendMathTolerance<TypeParam>());
      test::ExpectBackendClose(values[device], expected_value, {0.0F, 0.0F});
      std::vector<int32_t> actual_state;
      ASSERT_TRUE(state.Read(&actual_state));
      EXPECT_EQ(actual_state, (std::vector<int32_t>{47, 13, 29, kPosition}));
    }
    test::ExpectBackendClose(queries[0], queries[1], test::BackendMathTolerance<TypeParam>());
    test::ExpectBackendClose(keys[0], keys[1], test::BackendMathTolerance<TypeParam>());
    test::ExpectBackendClose(values[0], values[1], {0.0F, 0.0F});
  }
}

} // namespace
} // namespace omni_runtime::kernels
