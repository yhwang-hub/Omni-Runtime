#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/floaw_step_kernel/floaw_step_cuda_runtime.h"

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
  ASSERT_TRUE((runtime::FlowStepKernelRuntime<float>::CompileAndLaunch(
      "flow_step_float", {.options = {.grid_dim = dim3(8, 1, 1),
                                      .block_dim = dim3(256, 1, 1),
                                      .shared_memory_bytes = 0},
                          .state = state_device->device_address(),
                          .velocity = velocity_device->device_address(),
                          .dt = kTimeStepSeconds,
                          .element_count = kElementCount,
                          .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 1.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels

// Backend parity and validation checks exercise the operator runtime dispatch.
#include "kernels/backend_test_utils.h"
#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/floaw_step_kernel/flow_step_opencl_runtime.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class FlowStepBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(FlowStepBackendTest, test::BackendTypes);

TYPED_TEST(FlowStepBackendTest, PositiveNegativeAndZeroStepsSupportAliasedTailSlices) {
  constexpr size_t kCount = 517U;
  const auto state = test::BackendInput<TypeParam>(kCount, 2);
  const auto velocity = test::BackendInput<TypeParam>(kCount, 11);
  for (const float dt : {0.0F, 0.03125F, -0.125F}) {
    std::vector<float> expected(kCount);
    for (size_t index = 0U; index < kCount; ++index) {
      expected[index] = test::BackendRound<TypeParam>(
          std::fma(dt, test::BackendFloat(velocity[index]), test::BackendFloat(state[index])));
    }
    for (const int32_t alias : {0, 1, 2}) {
      std::array<std::vector<float>, 2U> results;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        SCOPED_TRACE(dt);
        SCOPED_TRACE(alias);
        test::BackendTensor<TypeParam> x, v, y;
        ASSERT_TRUE(x.Initialize(this->devices()[device], state));
        ASSERT_TRUE(v.Initialize(this->devices()[device], velocity));
        ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(kCount)));
        auto &output = alias == 1 ? x : alias == 2 ? v : y;
        ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                     backend::FlowStep{.state = x.view(),
                                                                       .velocity = v.view(),
                                                                       .dt = dt,
                                                                       .element_count = kCount,
                                                                       .output = output.view()}));
        ASSERT_TRUE(output.ReadFloats(&results[device]));
        test::ExpectBackendClose(results[device], expected, {0.0F, 0.0F});
      }
      test::ExpectBackendClose(results[0], results[1], {0.0F, 0.0F});
    }
  }
}

TEST(FlowStepValidationTest, RejectsEmptyAndNonfiniteSteps) {
  backend::OpenCLInvocation call;
  backend::FlowStep operation{};
  EXPECT_FALSE(runtime::OpenCLOperator<backend::FlowStep>::Make(operation, &call));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::FlowStep>::Make(operation, nullptr));
  operation.element_count = 1U;
  operation.dt = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(runtime::OpenCLOperator<backend::FlowStep>::Make(operation, &call));
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
