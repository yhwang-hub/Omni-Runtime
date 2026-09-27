#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr uint64_t kElementCount = 12288ULL;
constexpr uint64_t kIntermediateSize = 12288ULL;

class SwigluKernelTest : public test::KernelTestFixture {};

TEST_F(SwigluKernelTest, ActivatesQwen3VlIntermediateState) {
  std::vector<float> gate(kIntermediateSize);
  std::vector<float> up(kIntermediateSize);
  std::vector<float> expected(kIntermediateSize);
  for (uint64_t index = 0; index < kIntermediateSize; ++index) {
    const float gate_value = 0.001f * static_cast<float>(index) - 6.0f;
    const float up_value = 1.0f + 0.0001f * static_cast<float>(index);
    gate[index] = gate_value;
    up[index] = up_value;
    const float silu = gate_value / (1.0f + std::exp(-gate_value));
    expected[index] = silu * up_value;
  }
  const auto gate_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), gate);
  const auto up_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), up);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("swiglu_float", test::MakeSwigluFloatSource()));
  ASSERT_TRUE(Launch(dim3(16, 1, 1), dim3(256, 1, 1), 0, gate_device->device_address(),
                     up_device->device_address(), kElementCount, output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels
