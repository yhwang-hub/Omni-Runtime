#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class KvCacheBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(KvCacheBackendTest, test::BackendTypes);

TYPED_TEST(KvCacheBackendTest, StaticAndResidentPositionsStoreExactSlicesAndAdvanceAllAxes) {
  constexpr int32_t kTokens = 3;
  constexpr int32_t kWidth = 37;
  constexpr int32_t kCapacity = 7;
  const auto key = test::BackendInput<TypeParam>(kTokens * kWidth, 4);
  const auto value = test::BackendInput<TypeParam>(kTokens * kWidth, 8);
  const std::vector<TypeParam> initial(kCapacity * kWidth, test::BackendValue<TypeParam>(-3.0F));
  for (const bool resident_position : {false, true}) {
    for (const int32_t start : {2, 6}) {
      SCOPED_TRACE(::testing::Message() << start << ", resident=" << resident_position);
      auto expected_key = initial;
      auto expected_value = initial;
      for (int32_t token = 0; token < kTokens && start + token < kCapacity; ++token) {
        for (int32_t dimension = 0; dimension < kWidth; ++dimension) {
          expected_key[(start + token) * kWidth + dimension] = key[token * kWidth + dimension];
          expected_value[(start + token) * kWidth + dimension] = value[token * kWidth + dimension];
        }
      }
      std::array<std::vector<float>, 2U> keys;
      std::array<std::vector<float>, 2U> values;
      std::array<std::vector<int32_t>, 2U> states;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        test::BackendTensor<TypeParam> k;
        test::BackendTensor<TypeParam> v;
        test::BackendTensor<TypeParam> kc;
        test::BackendTensor<TypeParam> vc;
        test::BackendTensor<int32_t> state;
        ASSERT_TRUE(k.Initialize(this->devices()[device], key));
        ASSERT_TRUE(v.Initialize(this->devices()[device], value));
        ASSERT_TRUE(kc.Initialize(this->devices()[device], initial));
        ASSERT_TRUE(vc.Initialize(this->devices()[device], initial));
        ASSERT_TRUE(state.Initialize(this->devices()[device], {11, 7, 23, start}));
        const auto positions = state.view().Slice(0U, 3U * sizeof(int32_t));
        const auto cache_position = state.view().Slice(3U * sizeof(int32_t), sizeof(int32_t));
        ASSERT_TRUE(this->devices()[device]->Compute(
            test::BackendType<TypeParam>(),
            backend::StoreKvCache{.key = k.view(),
                                  .value = v.view(),
                                  .token_count = kTokens,
                                  .key_value_size = kWidth,
                                  .start_position = resident_position ? 0 : start,
                                  .cache_capacity = kCapacity,
                                  .key_cache = kc.view(),
                                  .value_cache = vc.view(),
                                  .cache_position =
                                      resident_position ? cache_position : backend::DeviceView{}}));
        ASSERT_TRUE(this->devices()[device]->Compute(
            test::BackendType<TypeParam>(),
            backend::AdvanceSequence{.positions = positions, .cache_position = cache_position}));
        ASSERT_TRUE(kc.ReadFloats(&keys[device]));
        ASSERT_TRUE(vc.ReadFloats(&values[device]));
        ASSERT_TRUE(state.Read(&states[device]));
        test::ExpectBackendClose(keys[device], test::BackendFloats(expected_key), {0.0F, 0.0F});
        test::ExpectBackendClose(values[device], test::BackendFloats(expected_value), {0.0F, 0.0F});
        EXPECT_EQ(states[device], (std::vector<int32_t>{12, 8, 24, start + 1}));
      }
      test::ExpectBackendClose(keys[0], keys[1], {0.0F, 0.0F});
      test::ExpectBackendClose(values[0], values[1], {0.0F, 0.0F});
      EXPECT_EQ(states[0], states[1]);
    }
  }
}

} // namespace
} // namespace omni_runtime::kernels
