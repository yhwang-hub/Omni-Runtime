#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

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
  ASSERT_TRUE(Compile("argmax_float", test::MakeArgmaxFloatSource()));
  ASSERT_TRUE(Launch(dim3(kRowCount, 1, 1), dim3(256, 1, 1), 0, logits_device->device_address(),
                     kRowCount, kVocabSize, output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<int32_t> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(actual == expected);
}
} // namespace
} // namespace omni_runtime::kernels
