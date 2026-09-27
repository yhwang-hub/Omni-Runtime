#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/adarms_kernel/adarms_cuda_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kRowCount = 8;
constexpr int32_t kHiddenSize = 4096;
constexpr float kEpsilon = 1.0e-6f;
constexpr std::size_t kElementCount = 32768U;

class AdarmsKernelTest : public test::KernelTestFixture {};

TEST_F(AdarmsKernelTest, NormalizesQwen3VlAdaptiveHiddenState) {
  std::vector<float> input(kElementCount);
  std::vector<float> scale(kHiddenSize);
  std::vector<float> shift(kHiddenSize);
  std::vector<float> expected(kElementCount);
  for (int32_t row = 0; row < kRowCount; ++row) {
    float square_mean = 0.0f;
    for (int32_t dimension = 0; dimension < kHiddenSize; ++dimension) {
      const int64_t index = static_cast<int64_t>(row) * kHiddenSize + dimension;
      const float value = 0.002f * static_cast<float>(dimension) - 4.0f;
      input[index] = value;
      square_mean += value * value;
    }
    square_mean /= static_cast<float>(kHiddenSize);
    const float inverse_rms = 1.0f / std::sqrt(square_mean + kEpsilon);
    for (int32_t dimension = 0; dimension < kHiddenSize; ++dimension) {
      const int64_t index = static_cast<int64_t>(row) * kHiddenSize + dimension;
      scale[dimension] = 0.0001f * static_cast<float>(dimension);
      shift[dimension] = 0.001f * static_cast<float>(dimension);
      expected[index] = input[index] * inverse_rms * (1.0f + scale[dimension]) + shift[dimension];
    }
  }
  const auto input_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto scale_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), scale);
  const auto shift_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), shift);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE((runtime::AdaRmsNormKernelRuntime<float>::CompileAndLaunch(
      "ada_rms_norm_float", {.options = {.grid_dim = dim3(kRowCount, 1, 1),
                                         .block_dim = dim3(256, 1, 1),
                                         .shared_memory_bytes = 0},
                             .input = input_device->device_address(),
                             .scale = scale_device->device_address(),
                             .shift = shift_device->device_address(),
                             .rows = kRowCount,
                             .hidden_size = kHiddenSize,
                             .epsilon = kEpsilon,
                             .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-4f));
}
} // namespace
} // namespace omni_runtime::kernels

// Backend parity and validation checks exercise the operator runtime dispatch.
#include "kernels/backend_test_utils.h"
#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/adarms_kernel/adarms_opencl_runtime.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class AdaRmsNormBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(AdaRmsNormBackendTest, test::BackendTypes);

TYPED_TEST(AdaRmsNormBackendTest, TailRowsBroadcastAndInPlaceKeepFp32Modulation) {
  constexpr int32_t kRows = 3;
  constexpr float kEpsilon = 1.0e-5F;
  for (const int32_t hidden : {37, 513}) {
    const auto input = test::BackendInput<TypeParam>(kRows * hidden, 3);
    const auto scale = test::BackendInput<TypeParam>(hidden, 5);
    const auto shift = test::BackendInput<TypeParam>(hidden, 9);
    std::vector<float> expected(input.size());
    for (int32_t row = 0; row < kRows; ++row) {
      double sum = 0.0;
      for (int32_t d = 0; d < hidden; ++d) {
        const float value = test::BackendFloat(input[row * hidden + d]);
        sum += double(value) * value;
      }
      const float inverse = 1.0F / std::sqrt(static_cast<float>(sum / hidden) + kEpsilon);
      for (int32_t d = 0; d < hidden; ++d) {
        const size_t index = row * hidden + d;
        expected[index] = test::BackendRound<TypeParam>(
            std::fma(test::BackendFloat(input[index]) * inverse,
                     1.0F + test::BackendFloat(scale[d]), test::BackendFloat(shift[d])));
      }
    }
    for (const bool alias : {false, true}) {
      std::array<std::vector<float>, 2U> results;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        SCOPED_TRACE(hidden);
        SCOPED_TRACE(alias);
        test::BackendTensor<TypeParam> x, s, b, y;
        ASSERT_TRUE(x.Initialize(this->devices()[device], input));
        ASSERT_TRUE(s.Initialize(this->devices()[device], scale));
        ASSERT_TRUE(b.Initialize(this->devices()[device], shift));
        ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(input.size())));
        auto &output = alias ? x : y;
        ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                     backend::AdaRmsNorm{.input = x.view(),
                                                                         .scale = s.view(),
                                                                         .shift = b.view(),
                                                                         .rows = kRows,
                                                                         .hidden_size = hidden,
                                                                         .epsilon = kEpsilon,
                                                                         .output = output.view()}));
        ASSERT_TRUE(output.ReadFloats(&results[device]));
        test::ExpectBackendClose(results[device], expected,
                                 test::BackendMathTolerance<TypeParam>());
      }
      test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
    }
  }
}

TEST(AdaRmsNormValidationTest, RejectsInvalidDimensionsAndEpsilon) {
  backend::OpenCLInvocation call;
  backend::AdaRmsNorm operation{};
  EXPECT_FALSE(runtime::OpenCLOperator<backend::AdaRmsNorm>::Make(operation, &call));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::AdaRmsNorm>::Make(operation, nullptr));
  operation.rows = 1;
  operation.hidden_size = 1;
  operation.epsilon = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(runtime::OpenCLOperator<backend::AdaRmsNorm>::Make(operation, &call));
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
