#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kHeadCount = 32;
constexpr int32_t kTokenCount = 8;
constexpr int32_t kHeadDim = 128;
constexpr std::size_t kElementCount = 32768U;

class MhaKernelTest : public test::KernelTestFixture {
protected:
  static void MakeAttentionInput(std::vector<float> *const tensor) {
    tensor->assign(kElementCount, 0.0f);
    for (std::size_t index = 0; index < kElementCount; ++index) {
      (*tensor)[index] = 0.0005f * static_cast<float>(index % kHeadDim) - 0.032f;
    }
  }
};

TEST_F(MhaKernelTest, AttendsQwen3VlTextTokens) {
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  MakeAttentionInput(&query);
  MakeAttentionInput(&key);
  MakeAttentionInput(&value);
  std::vector<float> expected;
  ASSERT_TRUE(test::ReferenceAttention(query, key, value, kHeadCount, kTokenCount, kHeadDim, true,
                                       &expected));
  const auto query_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), query);
  const auto key_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), key);
  const auto value_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), value);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("mha_float", test::MakeMhaFloatSource()));
  ASSERT_TRUE(Launch(dim3(kHeadCount * kTokenCount, 1, 1), dim3(128, 1, 1), 0,
                     query_device->device_address(), key_device->device_address(),
                     value_device->device_address(), kHeadCount, kTokenCount, kHeadDim, true,
                     output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels
