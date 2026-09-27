#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

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
  ASSERT_TRUE(Compile("store_kv_cache_float", test::MakeStoreKvCacheFloatSource()));
  ASSERT_TRUE(Launch(dim3(32, 1, 1), dim3(256, 1, 1), 0, key_device->device_address(),
                     value_device->device_address(), kTokenCount, kKeyValueSize, kStartPosition,
                     kCacheCapacity, key_cache_device->device_address(),
                     value_cache_device->device_address()));
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
