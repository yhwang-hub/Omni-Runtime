#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/kv_cache_kernel/kv_cache_cuda_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kTokenCount = 8;
constexpr int32_t kKeyValueSize = 1024;
constexpr int32_t kStartPosition = 4;
constexpr int32_t kCacheCapacity = 32;
constexpr std::size_t kCacheSize = 32768U;
constexpr std::size_t kUpdateSize = 8192U;

class KvCacheKernelTest : public test::KernelTestFixture {};

TEST_F(KvCacheKernelTest, StoresQwen3VlGroupedKeyValue) {
  std::vector<float> key(kUpdateSize);
  std::vector<float> value(kUpdateSize);
  std::vector<float> expected_key(kCacheSize, 999.0f);
  std::vector<float> expected_value(kCacheSize, 999.0f);
  for (std::size_t index = 0; index < kUpdateSize; ++index) {
    key[index] = 0.001f * static_cast<float>(index % kKeyValueSize) - 0.512f;
    value[index] = 0.0005f * static_cast<float>(index % kKeyValueSize) - 0.256f;
  }
  for (int32_t token = 0; token < kTokenCount; ++token) {
    const int32_t cache_token = kStartPosition + token;
    for (int32_t element = 0; element < kKeyValueSize; ++element) {
      const int64_t update_index = static_cast<int64_t>(token) * kKeyValueSize + element;
      const int64_t cache_index = static_cast<int64_t>(cache_token) * kKeyValueSize + element;
      expected_key[cache_index] = key[update_index];
      expected_value[cache_index] = value[update_index];
    }
  }
  const auto key_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), key);
  const auto value_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), value);
  const auto key_cache_device =
      memory::Buffer::CreateDeviceBufferFromVector(backend(), expected_key);
  const auto value_cache_device =
      memory::Buffer::CreateDeviceBufferFromVector(backend(), expected_value);
  ASSERT_TRUE((runtime::StoreKvCacheKernelRuntime<float>::CompileAndLaunch(
      "store_kv_cache_float", {.options = {.grid_dim = dim3(32, 1, 1),
                                           .block_dim = dim3(256, 1, 1),
                                           .shared_memory_bytes = 0},
                               .key = key_device->device_address(),
                               .value = value_device->device_address(),
                               .token_count = kTokenCount,
                               .key_value_size = kKeyValueSize,
                               .start_position = kStartPosition,
                               .cache_capacity = kCacheCapacity,
                               .key_cache = key_cache_device->device_address(),
                               .value_cache = value_cache_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual_key(kCacheSize);
  std::vector<float> actual_value(kCacheSize);
  ASSERT_TRUE(key_cache_device->CopyDataToVector(&actual_key));
  ASSERT_TRUE(value_cache_device->CopyDataToVector(&actual_value));
  EXPECT_TRUE(test::IsClose(actual_key, expected_key, 1.0e-6f) &&
              test::IsClose(actual_value, expected_value, 1.0e-6f));
}
} // namespace
} // namespace omni_runtime::kernels

// Backend parity and validation checks exercise the operator runtime dispatch.
#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class KvCacheBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(KvCacheBackendTest, test::BackendTypes);

class RoundedKvCacheBackendTest : public test::BackendParityFixture {};

TEST_F(RoundedKvCacheBackendTest, Fp32CacheRoundsValuesToFp16WhenRequested) {
  const std::vector<float> input = {0.33333334F, -0.0000333333F, 1.0006F, 512.75F};
  std::vector<float> expected(input.size());
  std::transform(input.begin(), input.end(), expected.begin(),
                 [](const float value) { return test::BackendRound<Float16>(value); });
  for (const auto &device : devices()) {
    test::BackendTensor<float> x;
    test::BackendTensor<float> key;
    test::BackendTensor<float> value;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(key.Initialize(device, std::vector<float>(input.size())));
    ASSERT_TRUE(value.Initialize(device, std::vector<float>(input.size())));
    ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                backend::StoreKvCache{.key = x.view(),
                                                      .value = x.view(),
                                                      .token_count = 1,
                                                      .key_value_size = 4,
                                                      .start_position = 0,
                                                      .cache_capacity = 1,
                                                      .key_cache = key.view(),
                                                      .value_cache = value.view(),
                                                      .is_fp16_cache = true}));
    std::vector<float> actual;
    ASSERT_TRUE(key.ReadFloats(&actual));
    test::ExpectBackendClose(actual, expected, {0.0F, 0.0F});
    ASSERT_TRUE(value.ReadFloats(&actual));
    test::ExpectBackendClose(actual, expected, {0.0F, 0.0F});
  }
}

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
} // namespace omni_runtime::kernels::backend_runtime_test
