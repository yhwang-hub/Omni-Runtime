#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/flash_attention_test_utils.h"
#include "kernels/flash_attention_kernel/flash_attention_cuda_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kHeadCount = 32;
constexpr int32_t kTokenCount = 8;
constexpr int32_t kHeadDim = 128;
constexpr std::size_t kElementCount = 32768U;

class FlashAttentionKernelTest : public test::KernelTestFixture {
protected:
  static void MakeAttentionInput(std::vector<float> *const tensor) {
    tensor->assign(kElementCount, 0.0f);
    for (std::size_t index = 0; index < kElementCount; ++index) {
      (*tensor)[index] = 0.001f * static_cast<float>(index % kHeadDim) - 0.064f;
    }
  }
};

template <typename T> class FlashAttentionV1Test : public test::KernelTestFixture {};
using ScalarAttentionTypes = ::testing::Types<float, Float16, BFloat16>;
TYPED_TEST_SUITE(FlashAttentionV1Test, ScalarAttentionTypes);

TYPED_TEST(FlashAttentionV1Test, MatchesCpuForGroupedHeadsCausalMasksAndTailTiles) {
  test::CheckFlashAttention<TypeParam, runtime::FlashAttentionKernelRuntime>(this->backend());
}

TEST_F(FlashAttentionKernelTest, AttendsQwen3VlVisionTokens) {
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  MakeAttentionInput(&query);
  MakeAttentionInput(&key);
  MakeAttentionInput(&value);
  std::vector<float> expected;
  ASSERT_TRUE(test::ReferenceAttention(query, key, value, kHeadCount, kTokenCount, kHeadDim, false,
                                       &expected));
  const auto query_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), query);
  const auto key_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), key);
  const auto value_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), value);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE((runtime::FlashAttentionKernelRuntime<float>::CompileAndLaunch(
      "flash_attention_float", {.options = {.grid_dim = dim3(kHeadCount * kTokenCount, 1, 1),
                                            .block_dim = dim3(128, 1, 1),
                                            .shared_memory_bytes = 0},
                                .query = query_device->device_address(),
                                .key = key_device->device_address(),
                                .value = value_device->device_address(),
                                .heads = kHeadCount,
                                .tokens = kTokenCount,
                                .head_dim = kHeadDim,
                                .is_causal = false,
                                .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels
