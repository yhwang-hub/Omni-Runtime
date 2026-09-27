#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr uint64_t kQwen3VlHiddenSize = 4096ULL;
constexpr uint64_t kActivationElementCount = 8ULL * kQwen3VlHiddenSize;

class AddKernelTest : public test::KernelTestFixture {};

TEST_F(AddKernelTest, ComputesQwen3VlActivationAdd) {
  std::vector<float> lhs(kActivationElementCount);
  std::vector<float> rhs(kActivationElementCount);
  std::vector<float> expected(kActivationElementCount);
  for (uint64_t index = 0; index < kActivationElementCount; ++index) {
    const float value = 0.01f * static_cast<float>(index % kQwen3VlHiddenSize);
    lhs[index] = value;
    rhs[index] = 0.25f * value;
    expected[index] = value + 0.25f * value;
  }
  const auto lhs_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), lhs);
  const auto rhs_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), rhs);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("add_float", test::MakeAddFloatSource()));
  ASSERT_TRUE(Launch(dim3(16, 1, 1), dim3(256, 1, 1), 0, lhs_device->device_address(),
                     rhs_device->device_address(), kActivationElementCount,
                     output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 1.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels
