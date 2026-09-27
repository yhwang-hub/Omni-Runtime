#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/gated_add_kernel/gated_add_cuda_runtime.h"

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
  ASSERT_TRUE((runtime::GatedAddKernelRuntime<float>::CompileAndLaunch(
      "gated_add_float", {.options = {.grid_dim = dim3(16, 1, 1),
                                      .block_dim = dim3(256, 1, 1),
                                      .shared_memory_bytes = 0},
                          .lhs = lhs_device->device_address(),
                          .rhs = rhs_device->device_address(),
                          .gate = gate_device->device_address(),
                          .element_count = kElementCount,
                          .hidden_size = kHiddenSize,
                          .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels

// Backend parity and validation checks exercise the operator runtime dispatch.
#include "kernels/backend_test_utils.h"
#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/gated_add_kernel/gated_add_opencl_runtime.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class GatedAddBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(GatedAddBackendTest, test::BackendTypes);

TYPED_TEST(GatedAddBackendTest, BroadcastGateAndAliasedPartialRowsMatchCpu) {
  constexpr size_t kCount = 517U;
  constexpr int32_t kHidden = 37;
  const auto lhs = test::BackendInput<TypeParam>(kCount, 1);
  const auto rhs = test::BackendInput<TypeParam>(kCount, 4);
  const auto gate = test::BackendInput<TypeParam>(kHidden, 8);
  std::vector<float> expected(kCount);
  for (size_t index = 0U; index < kCount; ++index) {
    expected[index] = test::BackendRound<TypeParam>(
        std::fma(test::BackendFloat(rhs[index]), test::BackendFloat(gate[index % kHidden]),
                 test::BackendFloat(lhs[index])));
  }
  for (const int32_t alias : {0, 1, 2}) {
    std::array<std::vector<float>, 2U> results;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
      SCOPED_TRACE(alias);
      test::BackendTensor<TypeParam> x, r, g, y;
      ASSERT_TRUE(x.Initialize(this->devices()[device], lhs));
      ASSERT_TRUE(r.Initialize(this->devices()[device], rhs));
      ASSERT_TRUE(g.Initialize(this->devices()[device], gate));
      ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(kCount)));
      auto &output = alias == 1 ? x : alias == 2 ? r : y;
      ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                   backend::GatedAdd{.lhs = x.view(),
                                                                     .rhs = r.view(),
                                                                     .gate = g.view(),
                                                                     .element_count = kCount,
                                                                     .hidden_size = kHidden,
                                                                     .output = output.view()}));
      ASSERT_TRUE(output.ReadFloats(&results[device]));
      test::ExpectBackendClose(results[device], expected, {0.0F, 0.0F});
    }
    test::ExpectBackendClose(results[0], results[1], {0.0F, 0.0F});
  }
}

TEST(GatedAddValidationTest, RejectsInvalidHiddenSizeAndMissingBuffers) {
  backend::OpenCLInvocation call;
  backend::GatedAdd operation{};
  operation.element_count = 1U;
  EXPECT_FALSE(runtime::OpenCLOperator<backend::GatedAdd>::Make(operation, &call));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::GatedAdd>::Make(operation, nullptr));
  operation.hidden_size = 1;
  EXPECT_FALSE(runtime::OpenCLOperator<backend::GatedAdd>::Make(operation, &call));
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
