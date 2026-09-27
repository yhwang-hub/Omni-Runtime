#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr uint64_t kElementCount = 32768ULL;
constexpr int32_t kHiddenSize = 4096;

class GatedAddKernelTest : public test::KernelTestFixture {};

TEST_F(GatedAddKernelTest, AddsQwen3VlGatedResidual) {
  std::vector<float> lhs(kElementCount);
  std::vector<float> rhs(kElementCount);
  std::vector<float> gate(kHiddenSize);
  std::vector<float> expected(kElementCount);
  for (uint64_t index = 0; index < kElementCount; ++index) {
    const int32_t dimension = static_cast<int32_t>(index % kHiddenSize);
    const float lhs_value = 0.001f * static_cast<float>(dimension);
    const float rhs_value = 0.0005f * static_cast<float>(dimension);
    lhs[index] = lhs_value;
    rhs[index] = rhs_value;
    gate[dimension] = 0.5f + 0.0001f * static_cast<float>(dimension);
    expected[index] = lhs_value + rhs_value * gate[dimension];
  }
  const auto lhs_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), lhs);
  const auto rhs_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), rhs);
  const auto gate_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), gate);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("gated_add_float", test::MakeGatedAddFloatSource()));
  ASSERT_TRUE(Launch(dim3(16, 1, 1), dim3(256, 1, 1), 0, lhs_device->device_address(),
                     rhs_device->device_address(), gate_device->device_address(), kElementCount,
                     kHiddenSize, output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels
