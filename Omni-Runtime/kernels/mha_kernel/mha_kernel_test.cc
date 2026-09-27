#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/mha_kernel/mha_kernel_runtime.h"

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
  ASSERT_TRUE((runtime::MhaKernelRuntime<float>::CompileAndLaunch(
      "mha_float", {.options = {.grid_dim = dim3(kHeadCount * kTokenCount, 1, 1),
                                .block_dim = dim3(128, 1, 1),
                                .shared_memory_bytes = 0},
                    .query = query_device->device_address(),
                    .key = key_device->device_address(),
                    .value = value_device->device_address(),
                    .heads = kHeadCount,
                    .tokens = kTokenCount,
                    .head_dim = kHeadDim,
                    .is_causal = true,
                    .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5f));
}

TEST_F(MhaKernelTest, HandlesHeadMajorTailTiles) {
  constexpr int32_t kHeads = 3;
  constexpr int32_t kTokens = 17;
  constexpr int32_t kDimensions = 77;
  std::vector<float> query(kHeads * kTokens * kDimensions);
  std::vector<float> key(query.size());
  std::vector<float> value(query.size());
  for (size_t index = 0U; index < query.size(); ++index) {
    query[index] = (static_cast<int32_t>(index % 31U) - 15) / 16.0F;
    key[index] = (static_cast<int32_t>(index % 29U) - 14) / 16.0F;
    value[index] = (static_cast<int32_t>(index % 37U) - 18) / 16.0F;
  }
  const auto q = memory::Buffer::CreateDeviceBufferFromVector(backend(), query);
  const auto k = memory::Buffer::CreateDeviceBufferFromVector(backend(), key);
  const auto v = memory::Buffer::CreateDeviceBufferFromVector(backend(), value);
  const auto output = memory::Buffer::CreateDeviceBuffer<float>(backend(), query.size());
  ASSERT_NE(q, nullptr);
  ASSERT_NE(k, nullptr);
  ASSERT_NE(v, nullptr);
  ASSERT_NE(output, nullptr);
  for (const bool is_causal : {false, true}) {
    std::vector<float> expected;
    ASSERT_TRUE(test::ReferenceAttention(query, key, value, kHeads, kTokens, kDimensions, is_causal,
                                         &expected));
    ASSERT_TRUE((runtime::MhaKernelRuntime<float>::CompileAndLaunch(
        "mha_tail", {.options = {},
                     .query = q->device_address(),
                     .key = k->device_address(),
                     .value = v->device_address(),
                     .heads = kHeads,
                     .tokens = kTokens,
                     .head_dim = kDimensions,
                     .is_causal = is_causal,
                     .output = output->device_address()})));
    std::vector<float> actual;
    ASSERT_TRUE(output->CopyDataToVector(&actual));
    EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5F));
  }
}
} // namespace
} // namespace omni_runtime::kernels
