#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/add_kernel/add_cuda_runtime.h"

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
  ASSERT_TRUE((runtime::AddKernelRuntime<float>::CompileAndLaunch(
      "add_float", {.options = {.grid_dim = dim3(16, 1, 1),
                                .block_dim = dim3(256, 1, 1),
                                .shared_memory_bytes = 0},
                    .lhs = lhs_device->device_address(),
                    .rhs = rhs_device->device_address(),
                    .element_count = kActivationElementCount,
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

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class AddBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(AddBackendTest, test::BackendTypes);

TYPED_TEST(AddBackendTest, TailSlicesAndAliasedOutputsMatchExactCpuSum) {
  constexpr size_t kCount = 517U;
  const auto lhs = test::BackendInput<TypeParam>(kCount, 1);
  const auto rhs = test::BackendInput<TypeParam>(kCount, 7);
  std::vector<float> expected(kCount);
  for (size_t index = 0U; index < kCount; ++index) {
    expected[index] = test::BackendRound<TypeParam>(test::BackendFloat(lhs[index]) +
                                                    test::BackendFloat(rhs[index]));
  }
  for (const int32_t alias : {0, 1, 2}) {
    SCOPED_TRACE(alias);
    std::array<std::vector<float>, 2U> results;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
      test::BackendTensor<TypeParam> x;
      test::BackendTensor<TypeParam> r;
      test::BackendTensor<TypeParam> y;
      ASSERT_TRUE(x.Initialize(this->devices()[device], lhs));
      ASSERT_TRUE(r.Initialize(this->devices()[device], rhs));
      ASSERT_TRUE(y.Initialize(this->devices()[device],
                               std::vector<TypeParam>(kCount, test::BackendValue<TypeParam>(-23))));
      auto &output = alias == 1 ? x : alias == 2 ? r : y;
      ASSERT_TRUE(this->devices()[device]->Compute(
          test::BackendType<TypeParam>(),
          backend::Add{
              .lhs = x.view(), .rhs = r.view(), .element_count = kCount, .output = output.view()}));
      ASSERT_TRUE(output.ReadFloats(&results[device]));
      test::ExpectBackendClose(results[device], expected, {0.0F, 0.0F});
    }
    test::ExpectBackendClose(results[0], results[1], {0.0F, 0.0F});
  }
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
