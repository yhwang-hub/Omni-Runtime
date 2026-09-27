#include <cstdint>
#include <memory>
#include <iostream>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/matmul_kernel/matmul_kernel_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kRows = 8;
constexpr int32_t kInnerSize = 4096;
constexpr int32_t kColumns = 4096;

class MatmulKernelTest : public test::KernelTestFixture {};

TEST_F(MatmulKernelTest, DISABLED_ProfilesBlackwellStreamingGemv) {
  const auto flush = memory::Buffer::CreateDeviceBuffer<uint8_t>(backend(), 64U * 1024U * 1024U);
  ASSERT_NE(flush, nullptr);
  backend::EventTimer timer;
  ASSERT_TRUE(timer.Initialize(backend()));
  for (const auto &shape : {std::pair{4096, 4096}, std::pair{4096, 1024}, std::pair{4096, 12288},
                            std::pair{12288, 4096}, std::pair{4096, 151936}}) {
    const int32_t inner = shape.first;
    const int32_t columns = shape.second;
    std::vector<Float16> input(inner);
    std::vector<Float16> weight(static_cast<size_t>(inner) * columns);
    for (int32_t d = 0; d < inner; ++d) {
      input[d] = test::LowPrecision<Float16>((d % 47 - 23) / 37.0F);
    }
    for (size_t i = 0; i < weight.size(); ++i) {
      weight[i] = test::LowPrecision<Float16>((static_cast<int32_t>(i % 67) - 33) / 131.0F);
    }
    const auto x = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
    const auto w = memory::Buffer::CreateDeviceBufferFromVector(backend(), weight);
    const auto y = memory::Buffer::CreateDeviceBuffer<Float16>(backend(), columns);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(w, nullptr);
    ASSERT_NE(y, nullptr);
    const auto launch = [&](const bool is_blackwell) {
      if (is_blackwell) {
        return inner == 4096
                   ? runtime::BlackwellGemvKernelRuntime<Float16, 4096>::CompileAndLaunch(
                         "blackwell_gemv_bench", x->device_address(), w->device_address(), columns,
                         y->device_address())
                   : runtime::BlackwellGemvKernelRuntime<Float16, 12288>::CompileAndLaunch(
                         "blackwell_gemv_bench", x->device_address(), w->device_address(), columns,
                         y->device_address());
      }
      return runtime::CudaKernelRuntime::CompileAndLaunch(
          "generic_gemv_bench",
          runtime::CudaKernelRuntime::MakeKernelSource<Float16, true>(
              "kernels/matmul_kernel/matmul_kernel.cuh",
              "omni_runtime::kernels::matmul_kernel::LinearGemvImpl"),
          {.grid_dim = {static_cast<uint32_t>((columns + 7) / 8), 1U, 1U},
           .block_dim = {256U, 1U, 1U},
           .shared_memory_bytes = 0},
          static_cast<const Float16 *>(x->device_address()),
          static_cast<const Float16 *>(w->device_address()), inner, columns,
          static_cast<Float16 *>(y->device_address()));
    };
    ASSERT_TRUE(launch(false));
    ASSERT_TRUE(launch(true));
    double totals_ms[2] = {};
    for (int32_t iteration = 0; iteration < 20; ++iteration) {
      for (const bool is_blackwell : {iteration % 2 != 0, iteration % 2 == 0}) {
        ASSERT_TRUE(flush->Fill(1U));
        ASSERT_TRUE(timer.Start());
        ASSERT_TRUE(launch(is_blackwell));
        double elapsed_ms = 0.0;
        ASSERT_TRUE(timer.Stop(&elapsed_ms));
        totals_ms[is_blackwell ? 1 : 0] += elapsed_ms;
      }
    }
    std::cout << "GEMV K=" << inner << " N=" << columns
              << " dispatch_plus_generic_ms=" << totals_ms[0] / 20.0
              << " dispatch_plus_Blackwell_ms=" << totals_ms[1] / 20.0 << std::endl;
  }
}

template <typename T> class LinearMatmulLowPrecisionTest : public test::KernelTestFixture {};
using LinearTypes = ::testing::Types<Float16, BFloat16>;
TYPED_TEST_SUITE(LinearMatmulLowPrecisionTest, LinearTypes);

TYPED_TEST(LinearMatmulLowPrecisionTest, BlackwellGemvPreservesRoundedReductionAndOutputTails) {
  for (const int32_t inner : {4096, 12288}) {
    constexpr int32_t kColumns = 19;
    std::vector<TypeParam> input(inner);
    std::vector<TypeParam> weight(static_cast<size_t>(inner) * kColumns);
    for (int32_t d = 0; d < inner; ++d) {
      input[d] = test::LowPrecision<TypeParam>((d % 47 - 23) / 37.0F);
    }
    for (size_t i = 0; i < weight.size(); ++i) {
      weight[i] = test::LowPrecision<TypeParam>((static_cast<int32_t>(i % 67) - 33) / 131.0F);
    }
    const auto x = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), input);
    const auto w = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), weight);
    const auto y = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), kColumns);
    const auto reference = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), kColumns);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(w, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(reference, nullptr);
    ASSERT_TRUE((runtime::CudaKernelRuntime::CompileAndLaunch(
        "gemv_reference",
        runtime::CudaKernelRuntime::MakeKernelSource<TypeParam, true>(
            "kernels/matmul_kernel/matmul_kernel.cuh",
            "omni_runtime::kernels::matmul_kernel::LinearGemvImpl"),
        {.grid_dim = {3U, 1U, 1U}, .block_dim = {256U, 1U, 1U}, .shared_memory_bytes = 0},
        static_cast<const TypeParam *>(x->device_address()),
        static_cast<const TypeParam *>(w->device_address()), inner, kColumns,
        static_cast<TypeParam *>(reference->device_address()))));
    if (inner == 4096) {
      ASSERT_TRUE((runtime::BlackwellGemvKernelRuntime<TypeParam, 4096>::CompileAndLaunch(
          "blackwell_gemv", x->device_address(), w->device_address(), kColumns,
          y->device_address())));
    } else {
      ASSERT_TRUE((runtime::BlackwellGemvKernelRuntime<TypeParam, 12288>::CompileAndLaunch(
          "blackwell_gemv", x->device_address(), w->device_address(), kColumns,
          y->device_address())));
    }
    std::vector<TypeParam> actual;
    std::vector<TypeParam> expected;
    ASSERT_TRUE(y->template CopyDataToVector<TypeParam>(&actual));
    ASSERT_TRUE(reference->template CopyDataToVector<TypeParam>(&expected));
    for (int32_t column = 0; column < kColumns; ++column) {
      EXPECT_EQ(test::HostFloat(actual[column]), test::HostFloat(expected[column]));
    }
  }
}

template <typename T> class MatmulTypedTest : public test::KernelTestFixture {};
using MatmulTypes = ::testing::Types<float, Float16, BFloat16>;
TYPED_TEST_SUITE(MatmulTypedTest, MatmulTypes);

TYPED_TEST(MatmulTypedTest, HandlesRegisterTileTailsAndBothWeightLayouts) {
  constexpr int32_t kInner = 37;
  constexpr int32_t kColumns = 65;
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
  std::vector<TypeParam> weight(kInner * kColumns + 1U);
  std::vector<TypeParam> transposed(weight.size());
  for (int32_t d = 0; d < kInner; ++d) {
    for (int32_t column = 0; column < kColumns; ++column) {
      const TypeParam value = convert((d % 13 - column % 7) / 32.0F);
      weight[d * kColumns + column + 1] = value;
      transposed[column * kInner + d + 1] = value;
    }
  }
  const auto w = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), weight);
  const auto wt = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), transposed);
  ASSERT_NE(w, nullptr);
  ASSERT_NE(wt, nullptr);
  for (const int32_t rows : {3, 17, 35}) {
    std::vector<TypeParam> input(rows * kInner + 1U);
    for (size_t index = 1U; index < input.size(); ++index) {
      input[index] = convert((static_cast<int32_t>(index % 17U) - 8) / 32.0F);
    }
    const auto x = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), input);
    const auto y = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), rows * kColumns);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    for (const bool is_transposed : {false, true}) {
      const auto input_address = static_cast<const TypeParam *>(x->device_address()) + 1;
      const auto weight_address =
          static_cast<const TypeParam *>((is_transposed ? wt : w)->device_address()) + 1;
      const typename runtime::MatmulKernelRuntime<TypeParam>::Args args = {.options = {},
                                                                           .lhs = input_address,
                                                                           .rhs = weight_address,
                                                                           .rows = rows,
                                                                           .inner = kInner,
                                                                           .columns = kColumns,
                                                                           .output =
                                                                               y->device_address()};
      if (is_transposed) {
        ASSERT_TRUE((runtime::LinearMatmulKernelRuntime<TypeParam>::CompileAndLaunch(
            "linear_tile_tail", {.options = args.options,
                                 .lhs = args.lhs,
                                 .rhs = args.rhs,
                                 .rows = args.rows,
                                 .inner = args.inner,
                                 .columns = args.columns,
                                 .output = args.output})));
      } else {
        ASSERT_TRUE(
            (runtime::MatmulKernelRuntime<TypeParam>::CompileAndLaunch("matmul_tile_tail", args)));
      }
      std::vector<TypeParam> actual;
      ASSERT_TRUE(y->template CopyDataToVector<TypeParam>(&actual));
      for (int32_t row = 0; row < rows; ++row) {
        for (int32_t column = 0; column < kColumns; ++column) {
          float sum = 0.0F;
          for (int32_t d = 0; d < kInner; ++d) {
            sum +=
                to_float(input[row * kInner + d + 1]) * to_float(weight[d * kColumns + column + 1]);
          }
          EXPECT_FLOAT_EQ(to_float(actual[row * kColumns + column]), to_float(convert(sum)));
        }
      }
    }
  }
}

TYPED_TEST(LinearMatmulLowPrecisionTest, HandlesVectorAndUnalignedReduction) {
  for (const int32_t inner : {1, 31, 32, 37, 40, 257}) {
    for (const int32_t offset : {0, 1}) {
      constexpr int32_t kOutputSize = 13;
      std::vector<TypeParam> input(inner + offset);
      std::vector<TypeParam> weight(inner * kOutputSize + offset);
      std::vector<float> expected(kOutputSize);
      for (int32_t index = 0; index < inner; ++index) {
        input[index + offset] = test::LowPrecision<TypeParam>((index % 7 - 3) / 32.0F);
      }
      for (int32_t row = 0; row < kOutputSize; ++row) {
        for (int32_t column = 0; column < inner; ++column) {
          const TypeParam value = test::LowPrecision<TypeParam>((column % 5 - row % 3) / 16.0F);
          weight[row * inner + column + offset] = value;
          expected[row] += test::HostFloat(input[column + offset]) * test::HostFloat(value);
        }
        expected[row] = test::HostFloat(test::LowPrecision<TypeParam>(expected[row]));
      }
      const auto input_device =
          memory::Buffer::CreateDeviceBufferFromVector(this->backend(), input);
      const auto weight_device =
          memory::Buffer::CreateDeviceBufferFromVector(this->backend(), weight);
      const auto output_device =
          memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), kOutputSize);
      ASSERT_TRUE((runtime::LinearMatmulKernelRuntime<TypeParam>::CompileAndLaunch(
          "linear_low_precision",
          {.options = {},
           .lhs = static_cast<const TypeParam *>(input_device->device_address()) + offset,
           .rhs = static_cast<const TypeParam *>(weight_device->device_address()) + offset,
           .rows = 1,
           .inner = inner,
           .columns = kOutputSize,
           .output = output_device->device_address()})));
      std::vector<TypeParam> output(kOutputSize);
      ASSERT_TRUE(output_device->CopyToHost(output.size() * sizeof(TypeParam), output.data()));
      for (int32_t row = 0; row < kOutputSize; ++row) {
        EXPECT_FLOAT_EQ(test::HostFloat(output[row]), expected[row]);
      }
    }
  }
}

TYPED_TEST(LinearMatmulLowPrecisionTest, BackendProjectsBatchedRowsWithFloatAccumulation) {
  const std::vector<float> input = {1, 2, 3, 4, -1, 0, 1, 2};
  const std::vector<float> weight = {1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1};
  const std::vector<float> expected = {5, 5, 10, 1, 1, 2};
  std::vector<TypeParam> typed_input(input.size());
  std::vector<TypeParam> typed_weight(weight.size());
  for (size_t index = 0U; index < input.size(); ++index) {
    typed_input[index] = test::LowPrecision<TypeParam>(input[index]);
  }
  for (size_t index = 0U; index < weight.size(); ++index) {
    typed_weight[index] = test::LowPrecision<TypeParam>(weight[index]);
  }
  const auto lhs = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), typed_input);
  const auto rhs = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), typed_weight);
  const auto output =
      memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), expected.size());
  const backend::ScalarType type = std::is_same_v<TypeParam, Float16>
                                       ? backend::ScalarType::kFloat16
                                       : backend::ScalarType::kBFloat16;
  ASSERT_TRUE(this->backend()->InitializeCompute());
  ASSERT_TRUE(
      this->backend()->LinearMatmul(type, lhs->view(), rhs->view(), 2, 4, 3, output->view()));
  std::vector<TypeParam> actual(expected.size());
  ASSERT_TRUE(output->CopyToHost(actual.size() * sizeof(TypeParam), actual.data()));
  for (size_t index = 0U; index < actual.size(); ++index) {
    EXPECT_FLOAT_EQ(test::HostFloat(actual[index]), expected[index]);
  }
}

TYPED_TEST(LinearMatmulLowPrecisionTest, ReusesTunedRowMajorPlansAcrossInputs) {
  constexpr int32_t kRows = 7;
  constexpr int32_t kInner = 24;
  constexpr int32_t kColumns = 16;
  std::vector<TypeParam> input(kRows * kInner);
  std::vector<TypeParam> weight(kColumns * kInner);
  for (size_t index = 0U; index < weight.size(); ++index) {
    weight[index] = test::LowPrecision<TypeParam>((static_cast<int32_t>(index % 13U) - 6) / 16.0F);
  }
  const auto lhs =
      memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), input.size() + 1U);
  const auto rhs =
      memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), weight.size() + 1U);
  const auto output =
      memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), kRows * kColumns + 1U);
  ASSERT_TRUE(this->backend()->InitializeCompute());
  const size_t workspace_bytes = this->backend()->compute_workspace_bytes();
  const backend::ScalarType type = std::is_same_v<TypeParam, Float16>
                                       ? backend::ScalarType::kFloat16
                                       : backend::ScalarType::kBFloat16;
  for (const int32_t offset : {0, 1}) {
    const auto input_view = memory::Buffer::CreateDeviceBufferView(
        lhs, offset * sizeof(TypeParam), input.size() * sizeof(TypeParam));
    const auto weight_view = memory::Buffer::CreateDeviceBufferView(
        rhs, offset * sizeof(TypeParam), weight.size() * sizeof(TypeParam));
    const auto output_view = memory::Buffer::CreateDeviceBufferView(
        output, offset * sizeof(TypeParam), kRows * kColumns * sizeof(TypeParam));
    ASSERT_TRUE(weight_view->CopyFromHost(weight.data(), weight.size() * sizeof(TypeParam)));
    for (int32_t iteration = 0; iteration < 2; ++iteration) {
      for (size_t index = 0U; index < input.size(); ++index) {
        input[index] = test::LowPrecision<TypeParam>(
            (static_cast<int32_t>(index % 11U) - 5 + iteration) / 32.0F);
      }
      ASSERT_TRUE(input_view->CopyFromHost(input.data(), input.size() * sizeof(TypeParam)));
      ASSERT_TRUE(this->backend()->LinearMatmul(type, input_view->view(), weight_view->view(),
                                                kRows, kInner, kColumns, output_view->view()));
      std::vector<TypeParam> actual(kRows * kColumns);
      ASSERT_TRUE(output_view->CopyToHost(actual.size() * sizeof(TypeParam), actual.data()));
      for (int32_t row = 0; row < kRows; ++row) {
        for (int32_t column = 0; column < kColumns; ++column) {
          float sum = 0.0F;
          for (int32_t d = 0; d < kInner; ++d) {
            sum += test::HostFloat(input[row * kInner + d]) *
                   test::HostFloat(weight[column * kInner + d]);
          }
          EXPECT_FLOAT_EQ(test::HostFloat(actual[row * kColumns + column]),
                          test::HostFloat(test::LowPrecision<TypeParam>(sum)));
        }
      }
      EXPECT_EQ(this->backend()->compute_workspace_bytes(), workspace_bytes);
    }
  }
}

TEST_F(MatmulKernelTest, ProjectsQwen3VlHiddenState) {
  std::vector<float> lhs(static_cast<std::size_t>(kRows) * kInnerSize);
  std::vector<float> rhs(static_cast<std::size_t>(kInnerSize) * kColumns);
  std::vector<float> expected(static_cast<std::size_t>(kRows) * kColumns);
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t inner = 0; inner < kInnerSize; ++inner) {
      lhs[static_cast<int64_t>(row) * kInnerSize + inner] =
          0.0001f * static_cast<float>(inner) + 0.01f * static_cast<float>(row);
    }
  }
  for (int32_t inner = 0; inner < kInnerSize; ++inner) {
    for (int32_t column = 0; column < kColumns; ++column) {
      rhs[static_cast<int64_t>(inner) * kColumns + column] =
          0.00001f * static_cast<float>(column) + 0.001f * static_cast<float>(inner);
    }
  }
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t column = 0; column < kColumns; ++column) {
      float value = 0.0f;
      for (int32_t inner = 0; inner < kInnerSize; ++inner) {
        value += lhs[static_cast<int64_t>(row) * kInnerSize + inner] *
                 rhs[static_cast<int64_t>(inner) * kColumns + column];
      }
      expected[static_cast<int64_t>(row) * kColumns + column] = value;
    }
  }
  const auto lhs_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), lhs);
  const auto rhs_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), rhs);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE((runtime::MatmulKernelRuntime<float>::CompileAndLaunch(
      "matmul_float", {.options = {.grid_dim = dim3((kColumns + 15) / 16, (kRows + 15) / 16, 1),
                                   .block_dim = dim3(16, 16, 1),
                                   .shared_memory_bytes = 0},
                       .lhs = lhs_device->device_address(),
                       .rhs = rhs_device->device_address(),
                       .rows = kRows,
                       .inner = kInnerSize,
                       .columns = kColumns,
                       .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-3f));
}
} // namespace
} // namespace omni_runtime::kernels
