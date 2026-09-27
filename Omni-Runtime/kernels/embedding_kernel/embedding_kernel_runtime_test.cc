#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/embedding_kernel/embedding_cuda_runtime.h"

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
  ASSERT_TRUE((runtime::EmbeddingKernelRuntime<float>::CompileAndLaunch(
      "embedding_float", {.options = {.grid_dim = dim3(kTokenCount, 1, 1),
                                      .block_dim = dim3(256, 1, 1),
                                      .shared_memory_bytes = 0},
                          .indices = indices_device->device_address(),
                          .weight = weight_device->device_address(),
                          .token_count = kTokenCount,
                          .hidden_size = kHiddenSize,
                          .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 1.0e-6f));
}
} // namespace
} // namespace omni_runtime::kernels

// Backend parity and validation checks exercise the operator runtime dispatch.
#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels::backend_runtime_test {
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
} // namespace omni_runtime::kernels::backend_runtime_test
