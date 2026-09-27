#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kTokenCount = 16;
constexpr int32_t kVocabSize = 151936;
constexpr int32_t kHiddenSize = 4096;

class EmbeddingKernelTest : public test::KernelTestFixture {};

TEST_F(EmbeddingKernelTest, GathersQwen3VlTokenEmbedding) {
  std::vector<int32_t> indices(kTokenCount);
  std::vector<float> weight(static_cast<std::size_t>(kTokenCount) * kHiddenSize);
  std::vector<float> expected(static_cast<std::size_t>(kTokenCount) * kHiddenSize);
  for (int32_t token = 0; token < kTokenCount; ++token) {
    indices[token] = token;
    for (int32_t dimension = 0; dimension < kHiddenSize; ++dimension) {
      const int64_t source_index = static_cast<int64_t>(token) * kHiddenSize + dimension;
      const int64_t output_index = static_cast<int64_t>(token) * kHiddenSize + dimension;
      weight[source_index] = 0.001f * static_cast<float>(dimension) + static_cast<float>(token);
      expected[output_index] = weight[source_index];
    }
  }
  const auto indices_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), indices);
  const auto weight_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), weight);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("embedding_float", test::MakeEmbeddingFloatSource()));
  ASSERT_TRUE(Launch(dim3(kTokenCount, 1, 1), dim3(256, 1, 1), 0, indices_device->device_address(),
                     weight_device->device_address(), kTokenCount, kHiddenSize,
                     output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 1.0e-6f));
}
} // namespace
} // namespace omni_runtime::kernels
