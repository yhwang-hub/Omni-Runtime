#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/argmax_kernel/argmax_cuda_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kRowCount = 4;
constexpr int32_t kVocabSize = 151936;

class ArgmaxKernelTest : public test::KernelTestFixture {};

TEST_F(ArgmaxKernelTest, SelectsQwen3VlVocabularyToken) {
  std::vector<float> logits(static_cast<std::size_t>(kRowCount) * kVocabSize);
  std::vector<int32_t> expected(kRowCount);
  for (int32_t row = 0; row < kRowCount; ++row) {
    int32_t best_index = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int32_t column = 0; column < kVocabSize; ++column) {
      const int64_t index = static_cast<int64_t>(row) * kVocabSize + column;
      logits[index] = std::cos(0.0001f * static_cast<float>(column + row));
      if (logits[index] > best_value) {
        best_value = logits[index];
        best_index = column;
      }
    }
    expected[row] = best_index;
  }
  const auto logits_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), logits);
  const auto output_device =
      memory::Buffer::CreateDeviceBuffer<int32_t>(backend(), expected.size());
  ASSERT_TRUE((runtime::ArgmaxKernelRuntime<float>::CompileAndLaunch(
      "argmax_float", {.options = {.grid_dim = dim3(kRowCount, 1, 1),
                                   .block_dim = dim3(256, 1, 1),
                                   .shared_memory_bytes = 0},
                       .logits = logits_device->device_address(),
                       .rows = kRowCount,
                       .columns = kVocabSize,
                       .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<int32_t> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(actual == expected);
}
} // namespace
} // namespace omni_runtime::kernels

// Backend parity and validation checks exercise the operator runtime dispatch.
#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class ArgmaxBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(ArgmaxBackendTest, test::BackendTypes);

TYPED_TEST(ArgmaxBackendTest, MultirowReductionTailAndTiesReturnFirstIndexExactly) {
  constexpr int32_t kRows = 3;
  for (const int32_t kColumns : {259, 65537, 248321}) {
    auto logits = test::BackendInput<TypeParam>(kRows * kColumns, 4);
    logits[257] = test::BackendValue<TypeParam>(5.0F);
    logits[258] = test::BackendValue<TypeParam>(5.0F);
    logits[kColumns] = test::BackendValue<TypeParam>(4.0F);
    logits[2 * kColumns - 1] = test::BackendValue<TypeParam>(4.0F);
    logits[2 * kColumns + 102] = test::BackendValue<TypeParam>(3.0F);
    const std::vector<int32_t> expected = {257, 0, 102};
    std::array<std::vector<int32_t>, 2U> results;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
      test::BackendTensor<TypeParam> x;
      test::BackendTensor<int32_t> y;
      ASSERT_TRUE(x.Initialize(this->devices()[device], logits));
      ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<int32_t>(kRows, -23)));
      ASSERT_TRUE(this->devices()[device]->Compute(
          test::BackendType<TypeParam>(),
          backend::Argmax{
              .logits = x.view(), .rows = kRows, .columns = kColumns, .output = y.view()}));
      ASSERT_TRUE(y.Read(&results[device]));
      EXPECT_EQ(results[device], expected);
    }
    EXPECT_EQ(results[0], results[1]);
  }
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
