#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/layernorm_kernel/layernorm_kernel_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kTokenCount = 8;
constexpr int32_t kHiddenSize = 4096;
constexpr float kEpsilon = 1.0e-6f;
constexpr std::size_t kElementCount = 32768U;

class LayernormKernelTest : public test::KernelTestFixture {};

template <typename T> class ResidualRmsNormTest : public test::KernelTestFixture {};
using ResidualTypes = ::testing::Types<float, Float16, BFloat16>;
TYPED_TEST_SUITE(ResidualRmsNormTest, ResidualTypes);

TYPED_TEST(ResidualRmsNormTest, PreservesSeparateAddAndNormalizationRounding) {
  ASSERT_TRUE(this->backend()->InitializeCompute());
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
  const backend::ScalarType type = std::is_same_v<TypeParam, float> ? backend::ScalarType::kFloat32
                                   : std::is_same_v<TypeParam, Float16>
                                       ? backend::ScalarType::kFloat16
                                       : backend::ScalarType::kBFloat16;
  for (const int32_t hidden_size : {37, 128, 4096}) {
    for (const int32_t rows : {1, 3}) {
      const size_t count = static_cast<size_t>(rows) * hidden_size;
      std::vector<TypeParam> input(count);
      std::vector<TypeParam> residual(count);
      std::vector<TypeParam> weights(hidden_size);
      for (size_t i = 0; i < count; ++i) {
        input[i] = convert((static_cast<int32_t>(i % 47) - 23) / 37.0F);
        residual[i] = convert((static_cast<int32_t>(i % 31) - 15) / 17.0F);
      }
      for (int32_t i = 0; i < hidden_size; ++i) {
        weights[i] = convert(0.9F + (i % 11) / 37.0F);
      }
      const auto x = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), input);
      const auto r = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), residual);
      const auto w = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), weights);
      const auto sum = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), count);
      const auto expected = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), count);
      const auto y = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), count);
      ASSERT_NE(x, nullptr);
      ASSERT_NE(r, nullptr);
      ASSERT_NE(w, nullptr);
      ASSERT_NE(sum, nullptr);
      ASSERT_NE(expected, nullptr);
      ASSERT_NE(y, nullptr);
      ASSERT_TRUE(this->backend()->Compute(type, backend::Add{.lhs = x->view(),
                                                              .rhs = r->view(),
                                                              .element_count = count,
                                                              .output = sum->view()}));
      ASSERT_TRUE(this->backend()->Compute(type, backend::RmsNorm{.input = sum->view(),
                                                                  .weight = w->view(),
                                                                  .rows = rows,
                                                                  .hidden_size = hidden_size,
                                                                  .epsilon = 1e-6F,
                                                                  .output = expected->view()}));
      std::vector<TypeParam> expected_values;
      std::vector<TypeParam> expected_sum;
      ASSERT_TRUE(expected->template CopyDataToVector<TypeParam>(&expected_values));
      ASSERT_TRUE(sum->template CopyDataToVector<TypeParam>(&expected_sum));
      for (const bool is_in_place : {false, true}) {
        const auto &destination = is_in_place ? r : sum;
        ASSERT_TRUE(this->backend()->Compute(
            type, backend::ResidualRmsNorm{.input = x->view(),
                                           .residual = r->view(),
                                           .weight = w->view(),
                                           .rows = rows,
                                           .hidden_size = hidden_size,
                                           .epsilon = 1e-6F,
                                           .residual_output = destination->view(),
                                           .output = y->view()}));
        std::vector<TypeParam> actual_values;
        std::vector<TypeParam> actual_sum;
        ASSERT_TRUE(y->template CopyDataToVector<TypeParam>(&actual_values));
        ASSERT_TRUE(destination->template CopyDataToVector<TypeParam>(&actual_sum));
        for (size_t i = 0; i < count; ++i) {
          EXPECT_EQ(to_float(actual_values[i]), to_float(expected_values[i]));
          EXPECT_EQ(to_float(actual_sum[i]), to_float(expected_sum[i]));
        }
      }
    }
  }
}

TEST_F(LayernormKernelTest, ReusesReductionScratchWithoutRacing) {
  constexpr int32_t kRows = 512;
  constexpr int32_t kWidth = 1152;
  std::vector<float> input(kRows * kWidth);
  std::vector<float> expected(input.size());
  std::vector<float> weight(kWidth, 1.0F);
  std::vector<float> bias(kWidth, 0.0F);
  for (int32_t row = 0; row < kRows; ++row) {
    double sum = 0.0;
    double square_sum = 0.0;
    for (int32_t column = 0; column < kWidth; ++column) {
      const float value = column == row % kWidth ? 12000.0F : (column % 19 - 9) / 8.0F;
      input[row * kWidth + column] = value;
      sum += value;
      square_sum += static_cast<double>(value) * value;
    }
    const double mean = sum / kWidth;
    const double inverse = 1.0 / std::sqrt(square_sum / kWidth - mean * mean + 1.0e-6);
    for (int32_t column = 0; column < kWidth; ++column) {
      expected[row * kWidth + column] = (input[row * kWidth + column] - mean) * inverse;
    }
  }
  const auto x = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto w = memory::Buffer::CreateDeviceBufferFromVector(backend(), weight);
  const auto b = memory::Buffer::CreateDeviceBufferFromVector(backend(), bias);
  const auto y = memory::Buffer::CreateDeviceBuffer<float>(backend(), input.size());
  std::vector<float> actual(input.size());
  for (int32_t iteration = 0; iteration < 20; ++iteration) {
    ASSERT_TRUE((runtime::LayernormKernelRuntime<float>::CompileAndLaunch(
        "layernorm_reduction_reuse", {.options = {.grid_dim = {kRows, 1U, 1U},
                                                  .block_dim = {256U, 1U, 1U},
                                                  .shared_memory_bytes = 0},
                                      .input = x->device_address(),
                                      .weight = w->device_address(),
                                      .bias = b->device_address(),
                                      .rows = kRows,
                                      .hidden_size = kWidth,
                                      .epsilon = 1.0e-6F,
                                      .output = y->device_address()})));
    ASSERT_TRUE(y->CopyToHost(actual.size() * sizeof(float), actual.data()));
    ASSERT_TRUE(test::IsClose(actual, expected, 1.0e-4F));
  }
}

TEST_F(LayernormKernelTest, NormalizesQwen3VlHiddenState) {
  std::vector<float> input(kElementCount);
  std::vector<float> weight(kHiddenSize);
  std::vector<float> expected(kElementCount);
  for (int32_t row = 0; row < kTokenCount; ++row) {
    float square_mean = 0.0f;
    for (int32_t dimension = 0; dimension < kHiddenSize; ++dimension) {
      const int64_t index = static_cast<int64_t>(row) * kHiddenSize + dimension;
      const float value = 0.001f * static_cast<float>(dimension) - 2.0f;
      input[index] = value;
      square_mean += value * value;
    }
    square_mean /= static_cast<float>(kHiddenSize);
    const float inverse_rms = 1.0f / std::sqrt(square_mean + kEpsilon);
    for (int32_t dimension = 0; dimension < kHiddenSize; ++dimension) {
      const int64_t index = static_cast<int64_t>(row) * kHiddenSize + dimension;
      weight[dimension] = 1.0f + 0.0001f * static_cast<float>(dimension);
      expected[index] = input[index] * inverse_rms * weight[dimension];
    }
  }
  const auto input_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto weight_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), weight);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE((runtime::RmsNormKernelRuntime<float>::CompileAndLaunch(
      "qwen3_vl_rmsnorm_float_test", {.options = {.grid_dim = dim3(kTokenCount, 1, 1),
                                                  .block_dim = dim3(256, 1, 1),
                                                  .shared_memory_bytes = 0},
                                      .input = input_device->device_address(),
                                      .weight = weight_device->device_address(),
                                      .rows = kTokenCount,
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
