#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/fused_ffn_kernel/fused_ffn_cuda_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kRows = 1;
constexpr int32_t kInputSize = 4096;
constexpr int32_t kIntermediateSize = 12288;
constexpr int32_t kOutputSize = 4096;

class FusedFfnKernelTest : public test::KernelTestFixture {
protected:
  static float GeluTanh(const float value) {
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    return 0.5f * value * (1.0f + std::tanh(kAlpha * (value + kBeta * value * value * value)));
  }
};

template <typename T> class FusedFfnTypedTest : public test::KernelTestFixture {};
using FfnTypes = ::testing::Types<float, Float16, BFloat16>;
TYPED_TEST_SUITE(FusedFfnTypedTest, FfnTypes);

TYPED_TEST(FusedFfnTypedTest, HandlesMultipleRowsAndNonWarpAlignedDimensions) {
  constexpr int32_t kBatch = 3;
  constexpr int32_t kInput = 37;
  constexpr int32_t kIntermediate = 67;
  constexpr int32_t kOutput = 19;
  const auto convert = [](const float value) {
    if constexpr (std::is_same_v<TypeParam, float>) {
      return value;
    } else {
      return test::LowPrecision<TypeParam>(value);
    }
  };
  const auto to_float = [](const TypeParam value) {
    if constexpr (std::is_same_v<TypeParam, float>) {
      return value;
    } else {
      return test::HostFloat(value);
    }
  };
  std::vector<TypeParam> input(kBatch * kInput);
  std::vector<TypeParam> first(kIntermediate * kInput);
  std::vector<TypeParam> first_bias(kIntermediate);
  std::vector<TypeParam> second(kOutput * kIntermediate);
  std::vector<TypeParam> second_bias(kOutput);
  for (auto *const values : {&input, &first, &first_bias, &second, &second_bias}) {
    for (size_t index = 0U; index < values->size(); ++index) {
      (*values)[index] = convert((static_cast<int32_t>(index % 17U) - 8) / 32.0F);
    }
  }
  std::vector<float> expected(kBatch * kOutput);
  for (int32_t row = 0; row < kBatch; ++row) {
    std::vector<double> intermediate(kIntermediate);
    for (int32_t neuron = 0; neuron < kIntermediate; ++neuron) {
      double sum = to_float(first_bias[neuron]);
      for (int32_t d = 0; d < kInput; ++d) {
        sum += static_cast<double>(to_float(input[row * kInput + d])) *
               to_float(first[neuron * kInput + d]);
      }
      intermediate[neuron] =
          0.5 * sum * (1.0 + std::tanh(0.7978845608028654 * (sum + 0.044715 * sum * sum * sum)));
    }
    for (int32_t neuron = 0; neuron < kOutput; ++neuron) {
      double sum = to_float(second_bias[neuron]);
      for (int32_t d = 0; d < kIntermediate; ++d) {
        sum += intermediate[d] * to_float(second[neuron * kIntermediate + d]);
      }
      expected[row * kOutput + neuron] = sum;
    }
  }
  const auto x = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), input);
  const auto w1 = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), first);
  const auto b1 = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), first_bias);
  const auto w2 = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), second);
  const auto b2 = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), second_bias);
  const auto y = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), expected.size());
  ASSERT_NE(x, nullptr);
  ASSERT_NE(w1, nullptr);
  ASSERT_NE(b1, nullptr);
  ASSERT_NE(w2, nullptr);
  ASSERT_NE(b2, nullptr);
  ASSERT_NE(y, nullptr);
  ASSERT_TRUE((runtime::FusedFfnKernelRuntime<TypeParam>::CompileAndLaunch(
      "fused_ffn_tail", {.options = {},
                         .input = x->device_address(),
                         .fc1_weight = w1->device_address(),
                         .fc1_bias = b1->device_address(),
                         .fc2_weight = w2->device_address(),
                         .fc2_bias = b2->device_address(),
                         .rows = kBatch,
                         .input_size = kInput,
                         .intermediate_size = kIntermediate,
                         .output_size = kOutput,
                         .output = y->device_address()})));
  std::vector<TypeParam> actual;
  ASSERT_TRUE(y->template CopyDataToVector<TypeParam>(&actual));
  constexpr float kTolerance = std::is_same_v<TypeParam, float>     ? 1.0e-5F
                               : std::is_same_v<TypeParam, Float16> ? 1.0e-3F
                                                                    : 8.0e-3F;
  for (size_t index = 0U; index < actual.size(); ++index) {
    EXPECT_NEAR(to_float(actual[index]), expected[index], kTolerance);
  }
}

TEST_F(FusedFfnKernelTest, TransformsQwen3VlFeedForward) {
  std::vector<float> input(kInputSize);
  std::vector<float> fc1_weight(static_cast<std::size_t>(kIntermediateSize) * kInputSize);
  std::vector<float> fc1_bias(kIntermediateSize);
  std::vector<float> fc2_weight(static_cast<std::size_t>(kOutputSize) * kIntermediateSize);
  std::vector<float> fc2_bias(kOutputSize);
  std::vector<float> expected(kOutputSize);
  for (int32_t index = 0; index < kInputSize; ++index) {
    input[index] = 0.0001f * static_cast<float>(index) - 0.2f;
  }
  for (int32_t neuron = 0; neuron < kIntermediateSize; ++neuron) {
    fc1_bias[neuron] = 0.00001f * static_cast<float>(neuron);
    for (int32_t index = 0; index < kInputSize; ++index) {
      fc1_weight[static_cast<int64_t>(neuron) * kInputSize + index] =
          0.000001f * static_cast<float>(index) + 0.0000001f * static_cast<float>(neuron);
    }
  }
  for (int32_t neuron = 0; neuron < kOutputSize; ++neuron) {
    fc2_bias[neuron] = 0.00001f * static_cast<float>(neuron);
    for (int32_t index = 0; index < kIntermediateSize; ++index) {
      fc2_weight[static_cast<int64_t>(neuron) * kIntermediateSize + index] =
          0.0000001f * static_cast<float>(index) + 0.00000001f * static_cast<float>(neuron);
    }
  }
  std::vector<float> intermediate(kIntermediateSize);
  for (int32_t neuron = 0; neuron < kIntermediateSize; ++neuron) {
    float value = fc1_bias[neuron];
    for (int32_t index = 0; index < kInputSize; ++index) {
      value += input[index] * fc1_weight[static_cast<int64_t>(neuron) * kInputSize + index];
    }
    intermediate[neuron] = GeluTanh(value);
  }
  for (int32_t neuron = 0; neuron < kOutputSize; ++neuron) {
    float value = fc2_bias[neuron];
    for (int32_t index = 0; index < kIntermediateSize; ++index) {
      value += intermediate[index] *
               fc2_weight[static_cast<int64_t>(neuron) * kIntermediateSize + index];
    }
    expected[neuron] = value;
  }
  const auto input_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto fc1_weight_device =
      memory::Buffer::CreateDeviceBufferFromVector(backend(), fc1_weight);
  const auto fc1_bias_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), fc1_bias);
  const auto fc2_weight_device =
      memory::Buffer::CreateDeviceBufferFromVector(backend(), fc2_weight);
  const auto fc2_bias_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), fc2_bias);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  const int shared_memory_bytes = (kInputSize + kIntermediateSize) * sizeof(float);
  ASSERT_TRUE((runtime::FusedFfnKernelRuntime<float>::CompileAndLaunch(
      "fused_ffn_float", {.options = {.grid_dim = dim3(kRows, 1, 1),
                                      .block_dim = dim3(256, 1, 1),
                                      .shared_memory_bytes = shared_memory_bytes},
                          .input = input_device->device_address(),
                          .fc1_weight = fc1_weight_device->device_address(),
                          .fc1_bias = fc1_bias_device->device_address(),
                          .fc2_weight = fc2_weight_device->device_address(),
                          .fc2_bias = fc2_bias_device->device_address(),
                          .rows = kRows,
                          .input_size = kInputSize,
                          .intermediate_size = kIntermediateSize,
                          .output_size = kOutputSize,
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
#include "kernels/fused_ffn_kernel/fused_ffn_opencl_runtime.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class FusedFfnBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(FusedFfnBackendTest, test::BackendTypes);

TYPED_TEST(FusedFfnBackendTest, TailNeuronsAndRowsKeepUnroundedGeluIntermediate) {
  constexpr int32_t kRows = 3;
  constexpr int32_t kInput = 37;
  constexpr int32_t kIntermediate = 67;
  constexpr int32_t kOutput = 19;
  auto input = test::BackendInput<TypeParam>(kRows * kInput, 1);
  auto first = test::BackendInput<TypeParam>(kIntermediate * kInput, 3);
  const auto first_bias = test::BackendInput<TypeParam>(kIntermediate, 5);
  auto second = test::BackendInput<TypeParam>(kOutput * kIntermediate, 7);
  const auto second_bias = test::BackendInput<TypeParam>(kOutput, 9);
  for (auto *const values : {&input, &first, &second}) {
    for (auto &value : *values) {
      value = test::BackendValue<TypeParam>(test::BackendFloat(value) * 0.25F);
    }
  }
  std::vector<float> expected(kRows * kOutput);
  bool rounded_intermediate_differs = false;
  for (int32_t row = 0; row < kRows; ++row) {
    std::vector<float> intermediate(kIntermediate);
    for (int32_t neuron = 0; neuron < kIntermediate; ++neuron) {
      double sum = test::BackendFloat(first_bias[neuron]);
      for (int32_t d = 0; d < kInput; ++d) {
        sum += double(test::BackendFloat(input[row * kInput + d])) *
               test::BackendFloat(first[neuron * kInput + d]);
      }
      const float value = static_cast<float>(sum);
      intermediate[neuron] = 0.5F * value *
                             (1.0F + std::tanh(0.7978845608028654F *
                                               std::fma(0.044715F * value * value, value, value)));
    }
    for (int32_t neuron = 0; neuron < kOutput; ++neuron) {
      double sum = test::BackendFloat(second_bias[neuron]);
      double rounded_sum = sum;
      for (int32_t d = 0; d < kIntermediate; ++d) {
        const float weight = test::BackendFloat(second[neuron * kIntermediate + d]);
        sum += double(intermediate[d]) * weight;
        rounded_sum += double(test::BackendRound<TypeParam>(intermediate[d])) * weight;
      }
      const float result = test::BackendRound<TypeParam>(static_cast<float>(sum));
      expected[row * kOutput + neuron] = result;
      rounded_intermediate_differs |=
          result != test::BackendRound<TypeParam>(static_cast<float>(rounded_sum));
    }
  }
  if constexpr (!std::is_same_v<TypeParam, float>) {
    EXPECT_TRUE(rounded_intermediate_differs);
  }
  std::array<std::vector<float>, 2U> results;
  for (size_t device = 0U; device < this->devices().size(); ++device) {
    SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
    test::BackendTensor<TypeParam> x, w1, b1, w2, b2, y;
    ASSERT_TRUE(x.Initialize(this->devices()[device], input));
    ASSERT_TRUE(w1.Initialize(this->devices()[device], first));
    ASSERT_TRUE(b1.Initialize(this->devices()[device], first_bias));
    ASSERT_TRUE(w2.Initialize(this->devices()[device], second));
    ASSERT_TRUE(b2.Initialize(this->devices()[device], second_bias));
    ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(expected.size())));
    ASSERT_TRUE(this->devices()[device]->Compute(
        test::BackendType<TypeParam>(), backend::FusedFfn{.input = x.view(),
                                                          .fc1_weight = w1.view(),
                                                          .fc1_bias = b1.view(),
                                                          .fc2_weight = w2.view(),
                                                          .fc2_bias = b2.view(),
                                                          .rows = kRows,
                                                          .input_size = kInput,
                                                          .intermediate_size = kIntermediate,
                                                          .output_size = kOutput,
                                                          .output = y.view()}));
    ASSERT_TRUE(y.ReadFloats(&results[device]));
    test::ExpectBackendClose(results[device], expected, test::BackendMathTolerance<TypeParam>());
  }
  test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
}

TEST(FusedFfnValidationTest, RejectsInvalidLocalMemoryBudgetAndMissingBuffers) {
  backend::OpenCLInvocation call;
  backend::FusedFfn operation{};
  EXPECT_FALSE(runtime::OpenCLOperator<backend::FusedFfn>::Make(operation, nullptr));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::FusedFfn>::Make(operation, &call));
  operation.rows = operation.output_size = 1;
  operation.input_size = operation.intermediate_size = std::numeric_limits<int32_t>::max();
  EXPECT_FALSE(runtime::OpenCLOperator<backend::FusedFfn>::Make(operation, &call));
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
