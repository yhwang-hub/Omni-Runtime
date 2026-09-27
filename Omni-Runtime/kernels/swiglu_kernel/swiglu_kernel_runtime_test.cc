#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/swiglu_kernel/swiglu_cuda_runtime.h"

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
  ASSERT_TRUE((runtime::SwigluKernelRuntime<float>::CompileAndLaunch(
      "swiglu_float", {.options = {.grid_dim = dim3(16, 1, 1),
                                   .block_dim = dim3(256, 1, 1),
                                   .shared_memory_bytes = 0},
                       .gate = gate_device->device_address(),
                       .up = up_device->device_address(),
                       .element_count = kElementCount,
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

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class SwiGLUBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(SwiGLUBackendTest, test::BackendTypes);

TYPED_TEST(SwiGLUBackendTest, TailSlicesAndAliasedGateMatchCpuActivation) {
  constexpr size_t kCount = 515U;
  const auto gate = test::BackendInput<TypeParam>(kCount, 2);
  const auto up = test::BackendInput<TypeParam>(kCount, 5);
  std::vector<float> expected(kCount);
  for (size_t index = 0U; index < kCount; ++index) {
    const float value = test::BackendFloat(gate[index]);
    const float silu = test::BackendRound<TypeParam>(value / (1.0F + std::exp(-value)));
    expected[index] = test::BackendRound<TypeParam>(silu * test::BackendFloat(up[index]));
  }
  for (const bool is_ggml_silu : {false, true}) {
    for (const bool in_place : {false, true}) {
      std::array<std::vector<float>, 2U> results;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        test::BackendTensor<TypeParam> g;
        test::BackendTensor<TypeParam> u;
        test::BackendTensor<TypeParam> y;
        ASSERT_TRUE(g.Initialize(this->devices()[device], gate));
        ASSERT_TRUE(u.Initialize(this->devices()[device], up));
        ASSERT_TRUE(
            y.Initialize(this->devices()[device],
                         std::vector<TypeParam>(kCount, test::BackendValue<TypeParam>(-23))));
        auto &output = in_place ? g : y;
        ASSERT_TRUE(this->devices()[device]->Compute(
            test::BackendType<TypeParam>(), backend::SwiGLU{.gate = g.view(),
                                                            .up = u.view(),
                                                            .element_count = kCount,
                                                            .output = output.view(),
                                                            .is_ggml_silu = is_ggml_silu}));
        ASSERT_TRUE(output.ReadFloats(&results[device]));
        test::ExpectBackendClose(results[device], expected,
                                 test::BackendMathTolerance<TypeParam>());
      }
      test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
      if (is_ggml_silu) {
        EXPECT_EQ(results[0], results[1]);
      }
    }
  }
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
