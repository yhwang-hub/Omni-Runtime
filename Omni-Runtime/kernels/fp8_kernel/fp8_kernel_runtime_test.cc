#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "kernels/backend_test_utils.h"
#include "kernels/fp8_kernel/fp8_runtime_utils.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

float DecodeFp8(const uint8_t code) {
  const int32_t magnitude = code & 127U;
  const int32_t exponent = magnitude >> 3;
  const int32_t mantissa = magnitude & 7;
  const float value =
      exponent == 0 ? mantissa / 512.0F : std::ldexp(1.0F + mantissa / 8.0F, exponent - 7);
  return (code & 128U) != 0U ? -value : value;
}

float QuantizeFp8(const float value) {
  const float magnitude = std::min(std::abs(value), 448.0F);
  int32_t best = 0;
  float error = magnitude;
  for (int32_t code = 1; code < 127; ++code) {
    const float distance = std::abs(magnitude - DecodeFp8(static_cast<uint8_t>(code)));
    if (distance < error || (distance == error && code % 2 == 0)) {
      error = distance;
      best = code;
    }
  }
  return std::copysign(DecodeFp8(static_cast<uint8_t>(best)), value);
}

class Fp8BackendTest : public test::BackendParityFixture {};

TEST_F(Fp8BackendTest, MatchesEveryEncodedPairAcrossDigitBoundaries) {
  constexpr int32_t kCodes = 256;
  constexpr int32_t kInner = 128;
  std::vector<float> input(kCodes * kInner, 0.0F);
  std::vector<uint8_t> weights(kCodes * kInner, 0U);
  for (int32_t code = 0; code < kCodes; ++code) {
    input[code * kInner + code % 127] = (code & 127) == 127
                                            ? std::numeric_limits<float>::quiet_NaN()
                                            : DecodeFp8(static_cast<uint8_t>(code));
    input[code * kInner + 127] = 448.0F;
    std::fill_n(weights.begin() + code * kInner, 127, static_cast<uint8_t>(code));
  }
  for (const auto &device : devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<float> x;
    test::BackendTensor<uint8_t> w;
    test::BackendTensor<float> s;
    test::BackendTensor<float> y;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(w.Initialize(device, weights));
    ASSERT_TRUE(s.Initialize(device, {1.0F, 1.0F}));
    ASSERT_TRUE(y.Initialize(device, std::vector<float>(kCodes * kCodes)));
    for (const int32_t rows : {kCodes, 8}) {
      SCOPED_TRACE(rows);
      for (int32_t first = 0; first < kCodes; first += rows) {
        auto input_view = x.view();
        input_view.offset_bytes += first * kInner * sizeof(float);
        input_view.size_bytes = rows * kInner * sizeof(float);
        auto output_view = y.view();
        output_view.offset_bytes += first * kCodes * sizeof(float);
        output_view.size_bytes = rows * kCodes * sizeof(float);
        ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                    backend::Fp8Linear{.input = input_view,
                                                       .weight = w.view(),
                                                       .weight_scales = s.view(),
                                                       .rows = rows,
                                                       .inner = kInner,
                                                       .columns = kCodes,
                                                       .output = output_view}));
      }
      std::vector<float> actual;
      ASSERT_TRUE(y.ReadFloats(&actual));
      for (int32_t row = 0; row < kCodes; ++row) {
        for (int32_t column = 0; column < kCodes; ++column) {
          const float value = actual[row * kCodes + column];
          if ((row & 127) == 127 || (column & 127) == 127) {
            EXPECT_TRUE(std::isnan(value)) << row << ", " << column;
          } else {
            EXPECT_EQ(value, DecodeFp8(static_cast<uint8_t>(row)) *
                                 DecodeFp8(static_cast<uint8_t>(column)))
                << row << ", " << column;
          }
        }
      }
    }
  }
}

TEST_F(Fp8BackendTest, PreservesLogicalReductionOrderForReorderedHeadBlocks) {
  constexpr int32_t kInner = 260 * 128;
  constexpr int32_t kColumns = 128;
  std::vector<uint8_t> weights(kColumns * kInner, 0U);
  for (int32_t column = 0; column < kColumns; ++column) {
    weights[column * kInner] = 0x38U;
    weights[column * kInner + 128] = 0xb8U;
    weights[column * kInner + kInner / 2] = 0x38U;
  }
  std::vector<float> scales(kInner / 128, 1.0F);
  scales[0] = 65536.0F;
  scales[1] = 65536.0F;
  scales[kInner / 256] = 1.0F / 448.0F;
  for (const int32_t rows : {1, 8, 9}) {
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      test::BackendTensor<float> x;
      test::BackendTensor<uint8_t> w;
      test::BackendTensor<float> s;
      test::BackendTensor<float> y;
      ASSERT_TRUE(x.Initialize(device, std::vector<float>(rows * kInner, 448.0F)));
      ASSERT_TRUE(w.Initialize(device, weights));
      ASSERT_TRUE(s.Initialize(device, scales));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(rows * kColumns)));
      for (const int32_t groups : {1, 2}) {
        SCOPED_TRACE(groups);
        const backend::Fp8Linear operation{.input = x.view(),
                                           .weight = w.view(),
                                           .weight_scales = s.view(),
                                           .rows = rows,
                                           .inner = kInner,
                                           .columns = kColumns,
                                           .output = y.view(),
                                           .block_reduction_groups = groups};
        ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, operation));
        std::vector<float> actual;
        ASSERT_TRUE(y.ReadFloats(&actual));
        EXPECT_EQ(actual, std::vector<float>(rows * kColumns, groups == 1 ? 1.0F : 0.0F));
        for (const int32_t invalid_groups : {0, -1, 7}) {
          auto invalid = operation;
          invalid.block_reduction_groups = invalid_groups;
          EXPECT_FALSE(device->Compute(backend::ScalarType::kFloat32, invalid));
        }
      }
    }
  }
}

TEST_F(Fp8BackendTest, PropagatesNanCodesWithoutConvertingThemToFiniteIntegers) {
  constexpr int32_t kInner = 128;
  constexpr int32_t kColumns = 128;
  std::vector<uint8_t> weights(kInner * kColumns, 0x38U);
  constexpr int32_t kNanColumns = 16;
  for (int32_t column = 0; column < kNanColumns; ++column) {
    const int32_t index = 16 * (column % 8) + column;
    weights[column * kInner + index] = column < kNanColumns / 2 ? 0x7fU : 0xffU;
  }
  for (const int32_t rows : {1, 2, 8, 9}) {
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      test::BackendTensor<float> x;
      test::BackendTensor<uint8_t> w;
      test::BackendTensor<float> s;
      test::BackendTensor<float> y;
      std::vector<float> input(rows * kInner, 448.0F);
      input.back() = std::numeric_limits<float>::quiet_NaN();
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(w.Initialize(device, weights));
      ASSERT_TRUE(s.Initialize(device, {1.0F}));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(rows * kColumns)));
      ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                  backend::Fp8Linear{.input = x.view(),
                                                     .weight = w.view(),
                                                     .weight_scales = s.view(),
                                                     .rows = rows,
                                                     .inner = kInner,
                                                     .columns = kColumns,
                                                     .output = y.view()}));
      std::vector<float> actual;
      ASSERT_TRUE(y.ReadFloats(&actual));
      for (int32_t row = 0; row < rows; ++row) {
        for (int32_t column = 0; column < kColumns; ++column) {
          if (row == rows - 1 || column < kNanColumns) {
            EXPECT_TRUE(std::isnan(actual[row * kColumns + column]));
          } else {
            EXPECT_EQ(actual[row * kColumns + column], 448.0F * kInner);
          }
        }
      }
    }
  }
}

TEST_F(Fp8BackendTest, PreservesSubgroupTailsAndNanIsolationAcrossReductionWindows) {
  constexpr int32_t kColumns = 128;
  for (const int32_t blocks : {3, 259}) {
    const int32_t inner = blocks * 128;
    std::vector<uint8_t> weights(kColumns * inner, 0U);
    std::vector<float> scales(blocks);
    for (int32_t block = 0; block < blocks; ++block) {
      scales[block] = (block % 5 + 1) * 0.003F;
      for (int32_t column = 0; column < kColumns; ++column) {
        weights[column * inner + block * 128] = block % 2 == 0 ? 0x38U : 0xb8U;
      }
    }
    weights[(blocks - 1) * 128] = 0x7fU;
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      SCOPED_TRACE(blocks);
      test::BackendTensor<float> x;
      test::BackendTensor<float> s;
      test::BackendTensor<float> y;
      const auto w = memory::Buffer::CreateDeviceBufferFromVector(device, weights);
      ASSERT_NE(w, nullptr);
      ASSERT_TRUE(x.Initialize(device, std::vector<float>(inner, 448.0F)));
      ASSERT_TRUE(s.Initialize(device, scales));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(kColumns)));
      for (const int32_t groups : {1, blocks == 3 ? 3 : 7}) {
        float expected = 0.0F;
        for (int32_t logical = 0; logical < blocks; ++logical) {
          const int32_t physical = logical % groups * (blocks / groups) + logical / groups;
          const float product = physical % 2 == 0 ? 448.0F : -448.0F;
          const volatile float scaled = product * scales[physical];
          expected += scaled;
        }
        ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                    backend::Fp8Linear{.input = x.view(),
                                                       .weight = w->view(),
                                                       .weight_scales = s.view(),
                                                       .rows = 1,
                                                       .inner = inner,
                                                       .columns = kColumns,
                                                       .output = y.view(),
                                                       .block_reduction_groups = groups}));
        std::vector<float> actual;
        ASSERT_TRUE(y.ReadFloats(&actual));
        EXPECT_TRUE(std::isnan(actual.front()));
        for (int32_t column = 1; column < kColumns; ++column) {
          EXPECT_EQ(actual[column], expected) << column << ", " << groups;
        }
      }
    }
  }
}

TEST_F(Fp8BackendTest, PreservesSmallProductsAcrossCancellationInGemvAndTiledPrefill) {
  constexpr int32_t kInner = 128;
  constexpr int32_t kColumns = 128;
  std::vector<uint8_t> weights(kInner * kColumns, 0U);
  for (int32_t column = 0; column < kColumns; ++column) {
    weights[column * kInner] = 0x7eU;
    weights[column * kInner + 32] = 0x01U;
    weights[column * kInner + 64] = 0xfeU;
  }
  for (const int32_t rows : {1, 9, 65}) {
    std::vector<float> input(rows * kInner, 0.0F);
    for (int32_t row = 0; row < rows; ++row) {
      input[row * kInner] = 448.0F;
      input[row * kInner + 32] = 1.0F / 512.0F;
      input[row * kInner + 64] = 448.0F;
    }
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      test::BackendTensor<float> x;
      test::BackendTensor<uint8_t> w;
      test::BackendTensor<float> s;
      test::BackendTensor<float> y;
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(w.Initialize(device, weights));
      ASSERT_TRUE(s.Initialize(device, {1.0F}));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(rows * kColumns)));
      ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                  backend::Fp8Linear{.input = x.view(),
                                                     .weight = w.view(),
                                                     .weight_scales = s.view(),
                                                     .rows = rows,
                                                     .inner = kInner,
                                                     .columns = kColumns,
                                                     .output = y.view()}));
      std::vector<float> actual;
      ASSERT_TRUE(y.ReadFloats(&actual));
      EXPECT_EQ(actual, std::vector<float>(rows * kColumns, 1.0F / 262144.0F));
    }
  }
}

TEST_F(Fp8BackendTest, Bfloat16BlockScalesAndCapturedPrefillMatchReference) {
  constexpr int32_t kInner = 256;
  constexpr int32_t kColumns = 256;
  const std::vector<float> scales = {0.003F, 0.007F, 0.013F, 0.017F};
  std::vector<uint8_t> weights(kColumns * kInner);
  for (size_t index = 0U; index < weights.size(); ++index) {
    weights[index] = (index * 17U + index / kInner) % 127U;
  }
  for (const int32_t rows : {1, 9}) {
    auto input = test::BackendInput<backend::BFloat16>(rows * kInner, 31);
    std::vector<float> expected(rows * kColumns);
    for (int32_t row = 0; row < rows; ++row) {
      for (int32_t block = 0; block < 2; ++block) {
        float maximum = 1e-12F;
        for (int32_t d = 0; d < 128; ++d) {
          maximum = std::max(maximum,
                             std::abs(test::BackendFloat(input[row * kInner + block * 128 + d])));
        }
        const float scale = maximum * (1.0F / 448.0F);
        for (int32_t column = 0; column < kColumns; ++column) {
          double partial = 0.0;
          for (int32_t d = 0; d < 128; ++d) {
            partial +=
                QuantizeFp8(test::BackendFloat(input[row * kInner + block * 128 + d]) / scale) *
                DecodeFp8(weights[column * kInner + block * 128 + d]);
          }
          expected[row * kColumns + column] +=
              (static_cast<float>(partial) * scale) * scales[column / 128 * 2 + block];
        }
      }
    }
    for (float &value : expected) {
      value = test::BackendRound<backend::BFloat16>(value);
    }
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      test::BackendTensor<backend::BFloat16> x;
      test::BackendTensor<uint8_t> w;
      test::BackendTensor<float> s;
      test::BackendTensor<backend::BFloat16> y;
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(w.Initialize(device, weights));
      ASSERT_TRUE(s.Initialize(device, scales));
      ASSERT_TRUE(y.Initialize(device, std::vector<backend::BFloat16>(expected.size())));
      const backend::Fp8Linear operation{.input = x.view(),
                                         .weight = w.view(),
                                         .weight_scales = s.view(),
                                         .rows = rows,
                                         .inner = kInner,
                                         .columns = kColumns,
                                         .output = y.view()};
      const auto output = y.view();
      const backend::Allocation readback{.data = output.data,
                                         .size_bytes = output.size_bytes,
                                         .memory_kind = backend::MemoryKind::kDeviceView,
                                         .offset_bytes = output.offset_bytes};
      std::unique_ptr<backend::ExecutionPlan> plan;
      ASSERT_TRUE(device->CreateExecutionPlan(backend::ScalarType::kBFloat16,
                                              std::vector<backend::ComputeOperation>{operation},
                                              readback, &plan));
      std::vector<backend::BFloat16> actual(expected.size());
      double elapsed_ms = 0.0;
      ASSERT_TRUE(plan->Run(actual.size() * sizeof(backend::BFloat16), actual.data(), &elapsed_ms));
      for (size_t index = 0U; index < actual.size(); ++index) {
        EXPECT_EQ(test::BackendFloat(actual[index]), expected[index]) << index;
      }
    }
  }
}

TEST_F(Fp8BackendTest, MatchesPytorchScalarReciprocalAtFp8Boundary) {
  for (const int32_t rows : {1, 9}) {
    std::vector<float> input(rows * 128, 0.0F);
    for (int32_t row = 0; row < rows; ++row) {
      input[row * 128] = 2.40625F;
      input[row * 128 + 1] = -0.2470703125F;
    }
    // PyTorch scalar division rounds the reciprocal first: -46 becomes -44, not -48.
    const float expected = 404.0F * 0.005371094215661287F;
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      test::BackendTensor<float> x;
      test::BackendTensor<uint8_t> w;
      test::BackendTensor<float> s;
      test::BackendTensor<float> y;
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(w.Initialize(device, std::vector<uint8_t>(128 * 128, 0x38U)));
      ASSERT_TRUE(s.Initialize(device, {1.0F}));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(rows * 128)));
      ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                  backend::Fp8Linear{.input = x.view(),
                                                     .weight = w.view(),
                                                     .weight_scales = s.view(),
                                                     .rows = rows,
                                                     .inner = 128,
                                                     .columns = 128,
                                                     .output = y.view()}));
      std::vector<float> actual;
      ASSERT_TRUE(y.ReadFloats(&actual));
      for (const float value : actual) {
        EXPECT_FLOAT_EQ(value, expected);
      }
    }
  }
}

TEST_F(Fp8BackendTest, ReplaysMultirowFp32Projection) {
  constexpr int32_t kRows = 3;
  constexpr int32_t kInner = 128;
  constexpr int32_t kColumns = 48;
  auto input = test::BackendInput<float>(kRows * kInner, 23);
  const auto weights = test::BackendInput<float>(kColumns * kInner, 37);
  for (const auto &device : devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<float> x;
    test::BackendTensor<float> w;
    test::BackendTensor<float> y;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(w.Initialize(device, weights));
    ASSERT_TRUE(y.Initialize(device, std::vector<float>(kRows * kColumns)));
    const backend::Linear operation{.input = x.view(),
                                    .weight = w.view(),
                                    .rows = kRows,
                                    .inner = kInner,
                                    .columns = kColumns,
                                    .output = y.view()};
    const auto output = y.view();
    const backend::Allocation readback{.data = output.data,
                                       .size_bytes = output.size_bytes,
                                       .memory_kind = backend::MemoryKind::kDeviceView,
                                       .offset_bytes = output.offset_bytes};
    std::unique_ptr<backend::ExecutionPlan> plan;
    ASSERT_TRUE(device->CreateExecutionPlan(backend::ScalarType::kFloat32,
                                            std::vector<backend::ComputeOperation>{operation},
                                            readback, &plan));
    for (int32_t replay = 0; replay < 2; ++replay) {
      std::vector<float> actual(kRows * kColumns);
      double elapsed_ms = 0.0;
      ASSERT_TRUE(plan->Run(actual.size() * sizeof(float), actual.data(), &elapsed_ms));
      for (int32_t row = 0; row < kRows; ++row) {
        for (int32_t column = 0; column < kColumns; ++column) {
          float expected = 0.0F;
          for (int32_t d = 0; d < kInner; ++d) {
            expected += input[row * kInner + d] * weights[column * kInner + d];
          }
          EXPECT_NEAR(actual[row * kColumns + column], expected, 1e-4F);
        }
      }
    }
  }
}

TEST_F(Fp8BackendTest, MatchesIndependentBlockReferenceAndReplays) {
  constexpr int32_t kInner = 256;
  constexpr int32_t kColumns = 256;
  const std::vector<float> scales = {0.003F, 0.007F, 0.013F, 0.017F};
  std::vector<uint8_t> weights(kColumns * kInner);
  for (size_t index = 0U; index < weights.size(); ++index) {
    const uint8_t magnitude = static_cast<uint8_t>((index * 17U + index / kInner) % 127U);
    weights[index] = magnitude | (index % 3U == 0U ? 128U : 0U);
  }
  for (const int32_t rows : {1, 3, 8, 9, 16, 17, 33, 65}) {
    SCOPED_TRACE(rows);
    std::vector<float> input = test::BackendInput<float>(rows * kInner, 31);
    std::fill_n(input.begin(), 128U, 0.0F);
    input[128U] = 448.0F;
    input[129U] = 1.0625F;
    input[130U] = -1.1875F;
    std::vector<float> quantized(input.size());
    std::vector<float> activation_scales(input.size() / 128U);
    for (size_t first = 0U; first < input.size(); first += 128U) {
      float maximum = 1e-12F;
      for (size_t d = 0U; d < 128U; ++d) {
        maximum = std::max(maximum, std::abs(input[first + d]));
      }
      const float scale = maximum * (1.0F / 448.0F);
      activation_scales[first / 128U] = scale;
      for (size_t d = 0U; d < 128U; ++d) {
        quantized[first + d] = QuantizeFp8(input[first + d] / scale);
      }
    }
    std::vector<float> expected(rows * kColumns, 0.0F);
    for (int32_t row = 0; row < rows; ++row) {
      for (int32_t column = 0; column < kColumns; ++column) {
        for (int32_t block = 0; block < 2; ++block) {
          double partial = 0.0;
          for (int32_t d = 0; d < 128; ++d) {
            partial += quantized[row * kInner + block * 128 + d] *
                       DecodeFp8(weights[column * kInner + block * 128 + d]);
          }
          const volatile float scaled =
              (static_cast<float>(partial) * activation_scales[row * 2 + block]) *
              scales[column / 128 * 2 + block];
          expected[row * kColumns + column] += scaled;
        }
      }
    }
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      test::BackendTensor<float> x;
      test::BackendTensor<uint8_t> w;
      test::BackendTensor<float> s;
      test::BackendTensor<float> y;
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(w.Initialize(device, weights));
      ASSERT_TRUE(s.Initialize(device, scales));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(expected.size())));
      const auto aligned_weight = memory::Buffer::CreateDeviceBufferFromVector(device, weights);
      std::vector<uint8_t> padded_weight(weights.size() + 8U);
      std::copy(weights.begin(), weights.end(), padded_weight.begin() + 8U);
      const auto half_aligned_weight =
          memory::Buffer::CreateDeviceBufferFromVector(device, padded_weight);
      ASSERT_NE(aligned_weight, nullptr);
      ASSERT_NE(half_aligned_weight, nullptr);
      for (const auto &weight_view : {w.view(), aligned_weight->view(),
                                     half_aligned_weight->view().Slice(8U, weights.size())}) {
        SCOPED_TRACE(weight_view.offset_bytes);
        const backend::Fp8Linear operation{.input = x.view(),
                                           .weight = weight_view,
                                           .weight_scales = s.view(),
                                           .rows = rows,
                                           .inner = kInner,
                                           .columns = kColumns,
                                           .output = y.view()};
        const auto output = y.view();
        const backend::Allocation readback{.data = output.data,
                                           .size_bytes = output.size_bytes,
                                           .memory_kind = backend::MemoryKind::kDeviceView,
                                           .offset_bytes = output.offset_bytes};
        std::unique_ptr<backend::ExecutionPlan> plan;
        ASSERT_TRUE(device->CreateExecutionPlan(backend::ScalarType::kFloat32,
                                                std::vector<backend::ComputeOperation>{operation},
                                                readback, &plan));
        for (int32_t replay = 0; replay < 2; ++replay) {
          if (replay == 1) {
            std::vector<float> updated = input;
            for (float &value : updated) {
              value = -value;
            }
            const auto view = x.view();
            const backend::Allocation destination{.data = view.data,
                                                  .size_bytes = view.size_bytes,
                                                  .memory_kind = backend::MemoryKind::kDeviceView,
                                                  .offset_bytes = view.offset_bytes};
            ASSERT_TRUE(device->CopyHostToDevice(updated.data(), updated.size() * sizeof(float),
                                                 destination, nullptr));
          }
          std::vector<float> actual(expected.size());
          double elapsed_ms = 0.0;
          ASSERT_TRUE(plan->Run(actual.size() * sizeof(float), actual.data(), &elapsed_ms));
          for (size_t index = 0U; index < expected.size(); ++index) {
            EXPECT_EQ(actual[index], replay == 0 ? expected[index] : -expected[index]) << index;
          }
        }
        auto invalid = operation;
        invalid.weight_scales = {};
        EXPECT_FALSE(device->Compute(backend::ScalarType::kFloat32, invalid));
        invalid = operation;
        invalid.inner = 129;
        EXPECT_FALSE(device->Compute(backend::ScalarType::kFloat32, invalid));
        ASSERT_TRUE(device->CopyHostToDevice(input.data(), input.size() * sizeof(float),
                                             {.data = x.view().data,
                                              .size_bytes = x.view().size_bytes,
                                              .memory_kind = backend::MemoryKind::kDeviceView,
                                              .offset_bytes = x.view().offset_bytes},
                                             nullptr));
      }
    }
  }
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
