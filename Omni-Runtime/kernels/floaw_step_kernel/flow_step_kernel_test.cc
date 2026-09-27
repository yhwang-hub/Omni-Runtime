#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr uint64_t kElementCount = 4096ULL;
constexpr float kTimeStepSeconds = 0.125f;

class FlowStepKernelTest : public test::KernelTestFixture {};

TEST_F(FlowStepKernelTest, AdvancesPi05ActionState) {
  std::vector<float> state(kElementCount);
  std::vector<float> velocity(kElementCount);
  std::vector<float> expected(kElementCount);
  for (uint64_t index = 0; index < kElementCount; ++index) {
    const float value = 0.01f * static_cast<float>(index % 1024) - 5.0f;
    state[index] = value;
    velocity[index] = 0.25f * value;
    expected[index] = value + kTimeStepSeconds * (0.25f * value);
  }
  const auto state_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), state);
  const auto velocity_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), velocity);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("flow_step_float", test::MakeFlowStepFloatSource()));
  ASSERT_TRUE(Launch(dim3(8, 1, 1), dim3(256, 1, 1), 0, state_device->device_address(),
                     velocity_device->device_address(), kTimeStepSeconds, kElementCount,
                     output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 1.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels
