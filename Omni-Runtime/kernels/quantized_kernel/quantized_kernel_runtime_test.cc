#include <array>
#include <cstring>

#include "kernels/backend_test_utils.h"
#include "kernels/quantized_kernel/quantized_cuda_runtime.h"
#include "kernels/quantized_kernel/quantized_opencl_runtime.h"
#include "kernels/quantized_kernel/quantized_runtime_utils.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

class QuantizedBackendTest : public test::BackendParityFixture {};

TEST_F(QuantizedBackendTest, GgmlNvfp4QuantizationMatchesReferencePackedGolden) {
  constexpr int32_t kRows = 3;
  constexpr int32_t kInner = 64;
  std::vector<float> input(kRows * kInner, 0.0F);
  std::fill_n(input.begin() + kInner, kInner, 1.0F);
  std::fill_n(input.begin() + 2 * kInner, kInner, -1.0F);
  std::vector<uint8_t> expected(kRows * 36U, 0U);
  for (size_t row = 1U; row < kRows; ++row) {
    std::fill_n(expected.begin() + row * 36U, 4U, 126U);
    std::fill_n(expected.begin() + row * 36U + 4U, 32U, row == 1U ? 0x77U : 0xFFU);
  }
  const uint32_t scale_bits = 0x39C30C31U;
  float reference_scale = 0.0F;
  std::memcpy(&reference_scale, &scale_bits, sizeof(reference_scale));
  const std::vector<float> expected_scales = {0.0F,
                                              0.0F,
                                              0.0F,
                                              0.0F,
                                              reference_scale,
                                              reference_scale,
                                              reference_scale,
                                              reference_scale,
                                              reference_scale,
                                              reference_scale,
                                              reference_scale,
                                              reference_scale};
  for (const auto &device : devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<float> x;
    test::BackendTensor<uint16_t> q;
    test::BackendTensor<float> s;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(q.Initialize(device, std::vector<uint16_t>(kRows * 18U)));
    ASSERT_TRUE(s.Initialize(device, std::vector<float>(kRows * kInner / 16)));
    ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                backend::QuantizeActivation{.input = x.view(),
                                                            .weight_type = 40,
                                                            .rows = kRows,
                                                            .columns = kInner,
                                                            .is_decode = false,
                                                            .output = q.view(),
                                                            .scales = s.view(),
                                                            .is_ggml_rounding = true,
                                                            .is_packed_nvfp4 = true}));
    std::vector<uint16_t> actual;
    std::vector<float> scales;
    ASSERT_TRUE(q.Read(&actual));
    ASSERT_TRUE(s.ReadFloats(&scales));
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), expected.size()), 0);
    EXPECT_EQ(scales, expected_scales);
  }
}

TEST_F(QuantizedBackendTest, GgmlQ8RoundingPreservesReferenceHalfwayCodes) {
  std::vector<float> input(96U, 0.0F);
  for (size_t row = 0U; row < 3U; ++row) {
    const float maximum = static_cast<float>(28U + row) / 32.0F;
    input[row * 32U] = maximum;
    input[row * 32U + 1U] = maximum * 0.5F;
    input[row * 32U + 2U] = -maximum * 0.5F;
    input[row * 32U + 3U] = -maximum;
  }
  for (const bool is_decode : {false, true}) {
    const std::array<float, 3U> halfway = is_decode ? std::array<float, 3U>{64.0F, 64.0F, 63.0F}
                                                    : std::array<float, 3U>{64.0F, 63.0F, 64.0F};
    std::vector<float> expected(input.size(), 0.0F);
    for (size_t row = 0U; row < 3U; ++row) {
      expected[row * 32U] = 127.0F;
      expected[row * 32U + 1U] = halfway[row];
      expected[row * 32U + 2U] = -halfway[row];
      expected[row * 32U + 3U] = -127.0F;
    }
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      test::BackendTensor<float> x;
      test::BackendTensor<float> q;
      test::BackendTensor<float> s;
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(q.Initialize(device, std::vector<float>(input.size())));
      ASSERT_TRUE(s.Initialize(device, std::vector<float>(3U)));
      ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                  backend::QuantizeActivation{.input = x.view(),
                                                              .weight_type = 8,
                                                              .rows = 3,
                                                              .columns = 32,
                                                              .is_decode = is_decode,
                                                              .output = q.view(),
                                                              .scales = s.view(),
                                                              .is_ggml_rounding = true}));
      std::vector<float> actual;
      ASSERT_TRUE(q.ReadFloats(&actual));
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST_F(QuantizedBackendTest, TiledQ8QuantizationPreservesIndependentBlockReference) {
  for (const int32_t inner : {32, 256, 288, 5120, 17408}) {
    for (const int32_t rows : {1, 3, 8, 9}) {
      SCOPED_TRACE(inner);
      SCOPED_TRACE(rows);
      auto input = test::BackendInput<float>(rows * inner, 43);
      for (size_t first = 0U; first < input.size(); first += 32U) {
        if (first / 32U % 11U == 0U) {
          std::fill_n(input.begin() + first, 32U, 0.0F);
        }
      }
      for (const auto &device : devices()) {
        SCOPED_TRACE(backend::BackendName(device->kind()));
        test::BackendTensor<float> x;
        test::BackendTensor<float> reference;
        test::BackendTensor<float> reference_scales;
        test::BackendTensor<int8_t> packed;
        test::BackendTensor<float> scales;
        ASSERT_TRUE(x.Initialize(device, input));
        ASSERT_TRUE(reference.Initialize(device, std::vector<float>(input.size())));
        ASSERT_TRUE(reference_scales.Initialize(device, std::vector<float>(input.size() / 32U)));
        ASSERT_TRUE(packed.Initialize(device, std::vector<int8_t>(input.size())));
        ASSERT_TRUE(scales.Initialize(device, std::vector<float>(input.size() / 32U)));
        for (const int32_t type : {8, 40}) {
          for (const bool is_decode : {false, true}) {
            if (type == 40 && !is_decode) {
              continue;
            }
            for (const bool is_ggml_rounding : {false, true}) {
              const backend::QuantizeActivation independent{
                  .input = x.view(),
                  .weight_type = type,
                  .rows = static_cast<int32_t>(input.size() / 32U),
                  .columns = 32,
                  .is_decode = is_decode,
                  .output = reference.view(),
                  .scales = reference_scales.view(),
                  .is_ggml_rounding = is_ggml_rounding};
              ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, independent));
              backend::QuantizeActivation tiled = independent;
              tiled.rows = rows;
              tiled.columns = inner;
              tiled.output = packed.view();
              tiled.scales = scales.view();
              tiled.is_packed_q8 = true;
              const auto output = packed.view();
              const backend::Allocation readback{.data = output.data,
                                                 .size_bytes = output.size_bytes,
                                                 .memory_kind = backend::MemoryKind::kDeviceView,
                                                 .offset_bytes = output.offset_bytes};
              std::unique_ptr<backend::ExecutionPlan> plan;
              ASSERT_TRUE(device->CreateExecutionPlan(backend::ScalarType::kFloat32,
                                                      std::vector<backend::ComputeOperation>{tiled},
                                                      readback, &plan));
              std::vector<float> expected;
              std::vector<float> expected_scales;
              ASSERT_TRUE(reference.Read(&expected));
              ASSERT_TRUE(reference_scales.Read(&expected_scales));
              for (int32_t replay = 0; replay < 2; ++replay) {
                std::vector<int8_t> actual(input.size());
                std::vector<float> actual_scales;
                double elapsed_ms = 0.0;
                ASSERT_TRUE(plan->Run(output.size_bytes, actual.data(), &elapsed_ms));
                ASSERT_TRUE(scales.Read(&actual_scales));
                EXPECT_EQ(test::BackendFloats(actual), expected);
                EXPECT_EQ(actual_scales, expected_scales);
              }
            }
          }
        }
      }
    }
  }
}

bool MakeWeights(const int32_t type, const int32_t count, std::vector<uint16_t> *const storage,
                 std::vector<float> *const values) {
  OMNI_RETURN_VAL_IF(storage == nullptr || values == nullptr || count % (type == 8 ? 32 : 64) != 0,
                     false);
  values->resize(count);
  if (type == 8) {
    std::vector<uint8_t> bytes(static_cast<size_t>(count) / 32U * 34U);
    for (int32_t block = 0; block < count / 32; ++block) {
      const uint16_t scale = test::BackendValue<Float16>(0.03125F).bits;
      std::memcpy(bytes.data() + block * 34, &scale, sizeof(scale));
      for (int32_t j = 0; j < 32; ++j) {
        const int8_t code = static_cast<int8_t>((block * 17 + j * 7) % 255 - 127);
        std::memcpy(bytes.data() + block * 34 + 2 + j, &code, sizeof(code));
        (*values)[block * 32 + j] = static_cast<float>(code) * 0.03125F;
      }
    }
    storage->resize(bytes.size() / 2U);
    std::memcpy(storage->data(), bytes.data(), bytes.size());
    return true;
  }
  OMNI_RETURN_VAL_IF(type != 40, false);
  std::vector<uint8_t> bytes(static_cast<size_t>(count) / 64U * 36U);
  constexpr int32_t kValues[] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  for (int32_t block = 0; block < count / 64; ++block) {
    for (int32_t sub = 0; sub < 4; ++sub) {
      const uint8_t scale = static_cast<uint8_t>(sub == 0 ? 1 : 40 + sub * 8);
      bytes[block * 36 + sub] = scale;
      const float factor =
          sub == 0 ? std::ldexp(1.0F, -10) : std::ldexp(1.0F, static_cast<int32_t>(scale >> 3) - 8);
      for (int32_t j = 0; j < 8; ++j) {
        const uint8_t first = static_cast<uint8_t>((block + j) % 16);
        const uint8_t second = static_cast<uint8_t>((block + j + 8) % 16);
        bytes[block * 36 + 4 + sub * 8 + j] = first | (second << 4U);
        (*values)[block * 64 + sub * 16 + j] = kValues[first] * factor;
        (*values)[block * 64 + sub * 16 + j + 8] = kValues[second] * factor;
      }
    }
  }
  storage->resize(bytes.size() / 2U);
  std::memcpy(storage->data(), bytes.data(), bytes.size());
  return true;
}

TEST_F(QuantizedBackendTest, GgufBlockLayoutEmbeddingAndLinearAtOffsets) {
  constexpr int32_t kInner = 192;
  constexpr int32_t kColumns = 67;
  for (const int32_t type : {8, 40}) {
    std::vector<uint16_t> packed;
    std::vector<float> weights;
    ASSERT_TRUE(MakeWeights(type, kInner * kColumns, &packed, &weights));
    for (const int32_t rows : {1, 3, 65}) {
      const auto input = test::BackendInput<float>(rows * kInner, 4);
      std::vector<float> expected(rows * kColumns);
      for (int32_t row = 0; row < rows; ++row) {
        for (int32_t column = 0; column < kColumns; ++column) {
          double sum = 0.0;
          for (int32_t d = 0; d < kInner; ++d) {
            sum += input[row * kInner + d] * weights[column * kInner + d];
          }
          expected[row * kColumns + column] = static_cast<float>(sum) * 0.25F;
        }
      }
      for (const auto &device : devices()) {
        SCOPED_TRACE(backend::BackendName(device->kind()));
        SCOPED_TRACE(type);
        SCOPED_TRACE(rows);
        test::BackendTensor<float> x;
        test::BackendTensor<uint16_t> w;
        test::BackendTensor<float> y;
        ASSERT_TRUE(x.Initialize(device, input));
        ASSERT_TRUE(w.Initialize(device, packed));
        ASSERT_TRUE(y.Initialize(device, std::vector<float>(rows * kColumns)));
        ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                    backend::QuantizedLinear{x.view(), w.view(), type, 0.25F, rows,
                                                             kInner, kColumns, y.view()}));
        std::vector<float> actual;
        ASSERT_TRUE(y.ReadFloats(&actual));
        test::ExpectBackendClose(actual, expected, {3.0e-4F, 1.0e-5F});
        test::BackendTensor<int32_t> ids;
        test::BackendTensor<float> embedding;
        ASSERT_TRUE(ids.Initialize(device, {0, 66, 5}));
        ASSERT_TRUE(embedding.Initialize(device, std::vector<float>(3 * kInner)));
        ASSERT_TRUE(
            device->Compute(backend::ScalarType::kFloat32,
                            backend::QuantizedEmbedding{ids.view(), w.view(), type, 3, kInner,
                                                        kColumns, embedding.view()}));
        ASSERT_TRUE(embedding.ReadFloats(&actual));
        std::vector<float> expected_embedding;
        for (const int32_t id : {0, 66, 5}) {
          expected_embedding.insert(expected_embedding.end(), weights.begin() + id * kInner,
                                    weights.begin() + (id + 1) * kInner);
        }
        test::ExpectBackendClose(actual, expected_embedding, {0.0F, 0.0F});
      }
    }
  }
}

TEST(QuantizedValidationTest, RejectsUnsupportedTypesAndTruncatedBlocks) {
  backend::QuantizedLinear args{};
  backend::OpenCLInvocation call;
  EXPECT_FALSE(runtime::OpenCLOperator<backend::QuantizedLinear>::Make(args, &call));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::QuantizedLinear>::Make(args, nullptr));
  float dummy = 0.0F;
  backend::DeviceView view{&dummy, 0U, 34U, backend::BackendKind::kCuda};
  EXPECT_TRUE(runtime::HasGgufWeight(view, 8, 32, 1));
  EXPECT_FALSE(runtime::HasGgufWeight(view, 8, 33, 1));
  EXPECT_FALSE(runtime::HasGgufWeight(view, 40, 64, 1));
  EXPECT_FALSE(runtime::HasGgufWeight(view, 39, 32, 1));
}

TEST(QuantizedValidationTest, RejectsInvalidPackedNvfp4Layouts) {
  std::array<float, 256U> storage{};
  const backend::DeviceView view{storage.data(), 0U, sizeof(storage), backend::BackendKind::kCuda};
  const backend::QuantizeActivation quantize{.input = view.Slice(0U, 256U),
                                             .weight_type = 40,
                                             .rows = 1,
                                             .columns = 64,
                                             .is_decode = false,
                                             .output = view.Slice(0U, 36U),
                                             .scales = view.Slice(0U, 16U),
                                             .is_ggml_rounding = true,
                                             .is_packed_nvfp4 = true};
  EXPECT_TRUE(runtime::ValidateQuantized(quantize, sizeof(float)));
  EXPECT_FALSE(runtime::ValidateQuantized(quantize, sizeof(Float16)));
  auto invalid_quantize = quantize;
  invalid_quantize.output = view.Slice(0U, 35U);
  EXPECT_FALSE(runtime::ValidateQuantized(invalid_quantize, sizeof(float)));
  invalid_quantize = quantize;
  invalid_quantize.input = view.Slice(0U, 255U);
  EXPECT_FALSE(runtime::ValidateQuantized(invalid_quantize, sizeof(float)));
  invalid_quantize = quantize;
  invalid_quantize.scales = view.Slice(0U, 15U);
  EXPECT_FALSE(runtime::ValidateQuantized(invalid_quantize, sizeof(float)));
  invalid_quantize = quantize;
  invalid_quantize.is_decode = true;
  EXPECT_FALSE(runtime::ValidateQuantized(invalid_quantize, sizeof(float)));
  invalid_quantize = quantize;
  invalid_quantize.weight_type = 8;
  EXPECT_FALSE(runtime::ValidateQuantized(invalid_quantize, sizeof(float)));
  invalid_quantize = quantize;
  invalid_quantize.columns = 32;
  EXPECT_FALSE(runtime::ValidateQuantized(invalid_quantize, sizeof(float)));
  const backend::QuantizedLinear linear{.input = quantize.output,
                                        .weight = view.Slice(0U, 36U),
                                        .weight_type = 40,
                                        .weight_scale = 1.0F,
                                        .rows = 1,
                                        .inner = 64,
                                        .columns = 1,
                                        .output = view.Slice(0U, 4U),
                                        .activation_scales = quantize.scales,
                                        .activation_block_size = 16,
                                        .is_packed_nvfp4 = true};
  EXPECT_TRUE(runtime::ValidateQuantized(linear, sizeof(float)));
  EXPECT_FALSE(runtime::ValidateQuantized(linear, sizeof(Float16)));
  auto invalid_linear = linear;
  invalid_linear.input = view.Slice(0U, 35U);
  EXPECT_FALSE(runtime::ValidateQuantized(invalid_linear, sizeof(float)));
  invalid_linear = linear;
  invalid_linear.activation_block_size = 32;
  EXPECT_FALSE(runtime::ValidateQuantized(invalid_linear, sizeof(float)));
  invalid_linear = linear;
  invalid_linear.activation_scales = {};
  EXPECT_FALSE(runtime::ValidateQuantized(invalid_linear, sizeof(float)));
  invalid_linear = linear;
  invalid_linear.weight_type = 8;
  EXPECT_FALSE(runtime::ValidateQuantized(invalid_linear, sizeof(float)));
}

TEST_F(QuantizedBackendTest, DenseGgufFloatHalfAndBfloatStorageMatchesReference) {
  constexpr int32_t kInner = 64;
  constexpr int32_t kColumns = 67;
  const auto weights = test::BackendInput<float>(kInner * kColumns, 9);
  for (const int32_t type : {0, 1, 30}) {
    std::vector<uint16_t> bits(weights.size());
    for (size_t index = 0U; index < weights.size(); ++index) {
      bits[index] = type == 1 ? test::BackendValue<Float16>(weights[index]).bits
                              : test::BackendValue<BFloat16>(weights[index]).bits;
    }
    for (const int32_t rows : {1, 5, 65, 129}) {
      const auto input = test::BackendInput<float>(rows * kInner, 4);
      std::vector<float> expected(rows * kColumns);
      for (int32_t row = 0; row < rows; ++row) {
        for (int32_t column = 0; column < kColumns; ++column) {
          for (int32_t d = 0; d < kInner; ++d) {
            expected[row * kColumns + column] +=
                input[row * kInner + d] * weights[column * kInner + d];
          }
        }
      }
      for (const auto &device : devices()) {
        test::BackendTensor<float> x;
        test::BackendTensor<float> wf;
        test::BackendTensor<uint16_t> ws;
        test::BackendTensor<float> y;
        ASSERT_TRUE(x.Initialize(device, input));
        ASSERT_TRUE(type == 0 ? wf.Initialize(device, weights) : ws.Initialize(device, bits));
        ASSERT_TRUE(y.Initialize(device, std::vector<float>(expected.size())));
        ASSERT_TRUE(device->Compute(
            backend::ScalarType::kFloat32,
            backend::QuantizedLinear{x.view(), type == 0 ? wf.view() : ws.view(), type, 1.0F, rows,
                                     kInner, kColumns, y.view()}));
        std::vector<float> actual;
        ASSERT_TRUE(y.ReadFloats(&actual));
        test::ExpectBackendClose(actual, expected, {2.0e-5F, 1.0e-5F});
      }
    }
  }
}

TEST_F(QuantizedBackendTest, BfloatDenseProductsPreserveHalfwayRounding) {
  constexpr int32_t kRows = 65;
  constexpr int32_t kInner = 64;
  constexpr int32_t kColumns = 67;
  std::vector<float> input(kRows * kInner);
  for (size_t index = 0U; index < input.size(); ++index) {
    const int32_t rounding_offset = index % 3U == 0U ? 1 : index % 3U == 1U ? 0 : -1;
    const int32_t base_bits = 0x3f000000 + static_cast<int32_t>(index % 256U) * 0x8000;
    const uint32_t bits = static_cast<uint32_t>(base_bits + rounding_offset);
    const uint32_t signed_bits = bits | (index % 2U == 0U ? 0U : 0x80000000U);
    std::memcpy(&input[index], &signed_bits, sizeof(signed_bits));
  }
  std::vector<uint16_t> weights(kInner * kColumns, 0U);
  std::vector<float> expected(kRows * kColumns);
  for (int32_t column = 0; column < kColumns; ++column) {
    const int32_t d = column % kInner;
    weights[column * kInner + d] = test::BackendValue<BFloat16>(1.0F).bits;
    for (int32_t row = 0; row < kRows; ++row) {
      expected[row * kColumns + column] =
          test::BackendFloat(test::BackendValue<BFloat16>(input[row * kInner + d]));
    }
  }
  for (const auto &device : devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<float> x;
    test::BackendTensor<uint16_t> w;
    test::BackendTensor<float> y;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(w.Initialize(device, weights));
    ASSERT_TRUE(y.Initialize(device, std::vector<float>(expected.size())));
    ASSERT_TRUE(device->Compute(
        backend::ScalarType::kFloat32,
        backend::QuantizedLinear{x.view(), w.view(), 30, 1.0F, kRows, kInner, kColumns, y.view()}));
    std::vector<float> actual;
    ASSERT_TRUE(y.ReadFloats(&actual));
    EXPECT_EQ(actual, expected);
  }
}

TEST_F(QuantizedBackendTest, ActivationQuantizationAndScaledProductsMatchAcrossBackends) {
  constexpr int32_t kInner = 192;
  constexpr int32_t kColumns = 67;
  for (const int32_t type : {8, 40}) {
    std::vector<uint16_t> packed;
    std::vector<float> weights;
    ASSERT_TRUE(MakeWeights(type, kInner * kColumns, &packed, &weights));
    for (const bool is_decode : {false, true}) {
      const int32_t block = type == 8 || is_decode ? 32 : 16;
      for (const int32_t rows : {1, 3, 8, 9, 65}) {
        auto input = test::BackendInput<float>(rows * kInner, 4);
        std::fill_n(input.begin(), 32U, 0.0F);
        std::array<std::vector<float>, 2U> codes;
        std::array<std::vector<float>, 2U> scales;
        for (size_t device_index = 0U; device_index < devices().size(); ++device_index) {
          const auto &device = devices()[device_index];
          SCOPED_TRACE(backend::BackendName(device->kind()));
          SCOPED_TRACE(type);
          SCOPED_TRACE(is_decode);
          SCOPED_TRACE(rows);
          test::BackendTensor<float> x;
          test::BackendTensor<uint16_t> w;
          test::BackendTensor<float> q;
          test::BackendTensor<float> s;
          test::BackendTensor<float> y;
          ASSERT_TRUE(x.Initialize(device, input));
          ASSERT_TRUE(w.Initialize(device, packed));
          ASSERT_TRUE(q.Initialize(device, std::vector<float>(input.size())));
          ASSERT_TRUE(s.Initialize(device, std::vector<float>(rows * (kInner / block))));
          ASSERT_TRUE(y.Initialize(device, std::vector<float>(rows * kColumns)));
          ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                      backend::QuantizeActivation{x.view(), type, rows, kInner,
                                                                  is_decode, q.view(), s.view()}));
          ASSERT_TRUE(q.ReadFloats(&codes[device_index]));
          ASSERT_TRUE(s.ReadFloats(&scales[device_index]));
          if (block == 32) {
            for (size_t first = 0U; first < input.size(); first += 32U) {
              float maximum = 0.0F;
              for (size_t j = 0U; j < 32U; ++j) {
                maximum = std::max(maximum, std::fabs(input[first + j]));
              }
              const float d = maximum / 127.0F;
              const float expected_scale = is_decode ? test::BackendRound<Float16>(d) : d;
              EXPECT_EQ(scales[device_index][first / 32U], expected_scale);
              for (size_t j = 0U; j < 32U; ++j) {
                const float code = maximum == 0.0F ? 0.0F
                                   : is_decode     ? std::round(input[first + j] / d)
                                               : std::round(input[first + j] * (127.0F / maximum));
                EXPECT_EQ(codes[device_index][first + j], code);
              }
            }
          } else {
            for (size_t index = 0U; index < input.size(); ++index) {
              const float reconstructed =
                  codes[device_index][index] * scales[device_index][index / block];
              EXPECT_NEAR(reconstructed, input[index], 0.16F);
            }
          }
          ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                      backend::QuantizedLinear{q.view(), w.view(), type, 0.25F,
                                                               rows, kInner, kColumns, y.view(),
                                                               s.view(), block}));
          std::vector<float> actual;
          ASSERT_TRUE(y.ReadFloats(&actual));
          std::vector<float> expected(rows * kColumns);
          for (int32_t row = 0; row < rows; ++row) {
            for (int32_t column = 0; column < kColumns; ++column) {
              double sum = 0.0;
              for (int32_t d = 0; d < kInner; ++d) {
                const size_t index = row * kInner + d;
                sum += codes[device_index][index] * scales[device_index][index / block] *
                       weights[column * kInner + d];
              }
              expected[row * kColumns + column] = static_cast<float>(sum) * 0.25F;
            }
          }
          test::ExpectBackendClose(actual, expected, {5.0e-5F, 2.0e-5F});
        }
        test::ExpectBackendClose(codes[0], codes[1], {0.0F, 0.0F});
        test::ExpectBackendClose(scales[0], scales[1], {1.0e-7F, 1.0e-6F});
      }
    }
  }
}

TEST_F(QuantizedBackendTest, PackedQ8CodesPreserveScaledProductsAndCapturedReplay) {
  constexpr int32_t kColumns = 257;
  for (const int32_t inner : {4096, 4128, 4160, 5120, 5184, 6144}) {
    SCOPED_TRACE(inner);
    for (const int32_t type : {8, 40}) {
      if (type == 40 && inner % 64 != 0) {
        continue;
      }
      std::vector<uint16_t> weights;
      std::vector<float> decoded_weights;
      ASSERT_TRUE(MakeWeights(type, inner * kColumns, &weights, &decoded_weights));
      for (const int32_t rows : {1, 4, 16, 65, 129}) {
        if (type == 40 && rows > 8) {
          continue;
        }
        const auto input = test::BackendInput<float>(rows * inner, 29);
        for (const auto &device : devices()) {
          SCOPED_TRACE(backend::BackendName(device->kind()));
          SCOPED_TRACE(type);
          SCOPED_TRACE(rows);
          test::BackendTensor<float> x;
          test::BackendTensor<uint16_t> w;
          test::BackendTensor<float> codes;
          test::BackendTensor<int8_t> packed;
          test::BackendTensor<float> scales;
          test::BackendTensor<float> packed_scales;
          test::BackendTensor<float> y;
          test::BackendTensor<float> packed_y;
          ASSERT_TRUE(x.Initialize(device, input));
          ASSERT_TRUE(w.Initialize(device, weights));
          const auto aligned_weights =
              memory::Buffer::CreateDeviceBufferFromVector(device, weights);
          ASSERT_NE(aligned_weights, nullptr);
          ASSERT_TRUE(codes.Initialize(device, std::vector<float>(input.size())));
          ASSERT_TRUE(packed.Initialize(device, std::vector<int8_t>(input.size())));
          ASSERT_TRUE(scales.Initialize(device, std::vector<float>(input.size() / 32U)));
          ASSERT_TRUE(packed_scales.Initialize(device, std::vector<float>(input.size() / 32U)));
          ASSERT_TRUE(y.Initialize(device, std::vector<float>(rows * kColumns)));
          ASSERT_TRUE(packed_y.Initialize(device, std::vector<float>(rows * kColumns)));
          backend::QuantizeActivation quantize{.input = x.view(),
                                               .weight_type = type,
                                               .rows = rows,
                                               .columns = inner,
                                               .is_decode = rows <= 8,
                                               .output = codes.view(),
                                               .scales = scales.view(),
                                               .is_ggml_rounding = true};
          ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, quantize));
          backend::QuantizedLinear linear{.input = codes.view(),
                                          .weight = w.view(),
                                          .weight_type = type,
                                          .weight_scale = 0.25F,
                                          .rows = rows,
                                          .inner = inner,
                                          .columns = kColumns,
                                          .output = y.view(),
                                          .activation_scales = scales.view(),
                                          .activation_block_size = 32,
                                          .is_ggml_split_k = true};
          ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, linear));
          quantize.output = packed.view();
          quantize.scales = packed_scales.view();
          quantize.is_packed_q8 = true;
          linear.input = packed.view();
          linear.weight = aligned_weights->view();
          linear.output = packed_y.view();
          linear.activation_scales = packed_scales.view();
          linear.is_packed_q8 = true;
          const auto view = packed_y.view();
          const backend::Allocation readback{.data = view.data,
                                             .size_bytes = view.size_bytes,
                                             .memory_kind = backend::MemoryKind::kDeviceView,
                                             .offset_bytes = view.offset_bytes};
          std::unique_ptr<backend::ExecutionPlan> plan;
          ASSERT_TRUE(device->CreateExecutionPlan(
              backend::ScalarType::kFloat32,
              std::vector<backend::ComputeOperation>{quantize, linear}, readback, &plan));
          std::vector<float> captured_output(rows * kColumns);
          for (int32_t replay = 0; replay < 2; ++replay) {
            double elapsed_ms = 0.0;
            ASSERT_TRUE(plan->Run(view.size_bytes, captured_output.data(), &elapsed_ms));
          }
          std::vector<float> expected_codes;
          std::vector<int8_t> actual_codes;
          std::vector<float> expected_scales;
          std::vector<float> actual_scales;
          std::vector<float> expected_output;
          std::vector<float> actual_output;
          ASSERT_TRUE(codes.Read(&expected_codes));
          ASSERT_TRUE(packed.Read(&actual_codes));
          ASSERT_TRUE(scales.Read(&expected_scales));
          ASSERT_TRUE(packed_scales.Read(&actual_scales));
          ASSERT_TRUE(y.Read(&expected_output));
          ASSERT_TRUE(packed_y.Read(&actual_output));
          EXPECT_EQ(test::BackendFloats(actual_codes), expected_codes);
          EXPECT_EQ(actual_scales, expected_scales);
          EXPECT_EQ(actual_output, expected_output);
          linear.activation_block_size = 16;
          EXPECT_FALSE(runtime::ValidateQuantized(linear, sizeof(float)));
          quantize.is_packed_nvfp4 = true;
          EXPECT_FALSE(runtime::ValidateQuantized(quantize, sizeof(float)));
        }
      }
    }
  }
}

TEST_F(QuantizedBackendTest, IndependentPackedProjectionsPreserveCapturedJoinsAndAliases) {
  constexpr int32_t kInner = 5120;
  constexpr int32_t kColumns = 49;
  constexpr size_t kOffsetBytes = 2U * sizeof(float);
  const auto input = test::BackendInput<float>(kInner, 31);
  for (const int32_t type : {8, 40}) {
    SCOPED_TRACE(type);
    std::vector<uint16_t> weights;
    std::vector<float> decoded_weights;
    ASSERT_TRUE(MakeWeights(type, kInner * kColumns, &weights, &decoded_weights));
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      test::BackendTensor<float> x;
      test::BackendTensor<int8_t> codes;
      test::BackendTensor<float> scales;
      std::array<test::BackendTensor<float>, 4U> outputs;
      ASSERT_TRUE(x.Initialize(device, input));
      const auto w = memory::Buffer::CreateDeviceBufferFromVector(device, weights);
      ASSERT_NE(w, nullptr);
      ASSERT_TRUE(codes.Initialize(device, std::vector<int8_t>(kInner)));
      ASSERT_TRUE(scales.Initialize(device, std::vector<float>(kInner / 32)));
      for (auto &output : outputs) {
        ASSERT_TRUE(output.Initialize(device, std::vector<float>(kColumns + 2)));
      }
      const backend::QuantizeActivation quantize{.input = x.view(),
                                                 .weight_type = type,
                                                 .rows = 1,
                                                 .columns = kInner,
                                                 .is_decode = true,
                                                 .output = codes.view(),
                                                 .scales = scales.view(),
                                                 .is_ggml_rounding = true,
                                                 .is_packed_q8 = true};
      for (const int32_t scenario : {0, 1, 2}) {
        SCOPED_TRACE(scenario);
        ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, quantize));
        std::array<backend::DeviceView, 4U> views;
        for (size_t index = 0U; index < views.size(); ++index) {
          views[index] = outputs[index].view().Slice(kOffsetBytes, kColumns * sizeof(float));
        }
        if (scenario == 2) {
          views[1] = views[0];
          if (device->kind() == backend::BackendKind::kCuda) {
            views[1].data = static_cast<uint8_t *>(views[1].data) + views[1].offset_bytes;
            views[1].offset_bytes = 0U;
          }
        }
        std::vector<backend::ComputeOperation> operations;
        if (scenario == 1) {
          operations.emplace_back(quantize);
        }
        for (size_t index = 0U; index < views.size(); ++index) {
          operations.emplace_back(
              backend::QuantizedLinear{.input = codes.view(),
                                       .weight = w->view(),
                                       .weight_type = type,
                                       .weight_scale = static_cast<float>(index + 1U) * 0.25F,
                                       .rows = 1,
                                       .inner = kInner,
                                       .columns = kColumns,
                                       .output = views[index],
                                       .activation_scales = scales.view(),
                                       .activation_block_size = 32,
                                       .is_ggml_split_k = true,
                                       .is_packed_q8 = true});
        }
        if (scenario != 2) {
          for (size_t index = 1U; index < views.size(); ++index) {
            operations.emplace_back(backend::Add{.lhs = views[0],
                                                 .rhs = views[index],
                                                 .element_count = kColumns,
                                                 .output = views[0]});
          }
        }
        for (const auto &operation : operations) {
          ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, operation));
        }
        const backend::Allocation readback{.data = views[0].data,
                                           .size_bytes = views[0].size_bytes,
                                           .memory_kind = backend::MemoryKind::kDeviceView,
                                           .offset_bytes = views[0].offset_bytes};
        std::vector<float> expected(kColumns);
        ASSERT_TRUE(
            device->CopyDeviceToHost(readback, readback.size_bytes, expected.data(), nullptr));
        std::unique_ptr<backend::ExecutionPlan> plan;
        ASSERT_TRUE(device->CreateExecutionPlan(backend::ScalarType::kFloat32, operations, readback,
                                                &plan));
        for (int32_t replay = 0; replay < 3; ++replay) {
          std::vector<float> actual(kColumns);
          double elapsed_ms = 0.0;
          ASSERT_TRUE(plan->Run(readback.size_bytes, actual.data(), &elapsed_ms));
          EXPECT_EQ(actual, expected);
        }
      }
    }
  }
}

TEST_F(QuantizedBackendTest, PackedNvfp4RoundsScaledK16GroupsOnlyAtK64Boundary) {
  constexpr int32_t kRows = 16;
  constexpr int32_t kColumns = 8;
  constexpr int32_t kInner = 64;
  std::vector<uint8_t> input(kRows * 36);
  std::vector<uint8_t> weight(kColumns * 36);
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t sub = 0; sub < 4; ++sub) {
      input[row * 36 + sub] = sub % 2 == 0 ? 80 : 1;
      input[row * 36 + 4 + sub * 8] = 1;
    }
  }
  for (int32_t column = 0; column < kColumns; ++column) {
    for (int32_t sub = 0; sub < 4; ++sub) {
      weight[column * 36 + sub] = sub % 2 == 0 ? 80 : 1;
      weight[column * 36 + 4 + sub * 8] = sub == 2 ? 9 : 1;
    }
  }
  for (const auto &device : devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<uint8_t> x;
    test::BackendTensor<uint8_t> w;
    test::BackendTensor<float> s;
    test::BackendTensor<float> y;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(w.Initialize(device, weight));
    ASSERT_TRUE(s.Initialize(device, std::vector<float>(kRows * kInner / 16, 1.0F)));
    ASSERT_TRUE(y.Initialize(device, std::vector<float>(kRows * kColumns)));
    ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                backend::QuantizedLinear{.input = x.view(),
                                                         .weight = w.view(),
                                                         .weight_type = 40,
                                                         .weight_scale = 1.0F,
                                                         .rows = kRows,
                                                         .inner = kInner,
                                                         .columns = kColumns,
                                                         .output = y.view(),
                                                         .activation_scales = s.view(),
                                                         .activation_block_size = 16,
                                                         .is_packed_nvfp4 = true,
                                                         .is_ggml_split_k = true}));
    std::vector<float> actual;
    ASSERT_TRUE(y.ReadFloats(&actual));
    EXPECT_EQ(actual, std::vector<float>(kRows * kColumns, std::ldexp(1.0F, -19)));
  }
}

TEST_F(QuantizedBackendTest, PackedNvfp4K64TruncationMatchesExactScaledDots) {
  constexpr int32_t kRows = 48;
  constexpr int32_t kColumns = 257;
  constexpr int32_t kInner = 64;
  std::vector<uint8_t> input(kRows * 36);
  std::vector<uint8_t> weight(kColumns * 36);
  uint32_t state = 47U;
  for (auto *const storage : {&input, &weight}) {
    for (size_t index = 0U; index < storage->size(); ++index) {
      state = state * 1664525U + 1013904223U;
      (*storage)[index] = index % 36U < 4U ? (state >> 16U) % 126U + 1U : state >> 24U;
    }
  }
  for (int32_t sub = 0; sub < 4; ++sub) {
    input[sub] = 126U;
    weight[sub] = 126U;
    weight[36 + sub] = 126U;
  }
  for (int32_t index = 4; index < 36; ++index) {
    input[index] = 0x77U;
    weight[index] = 0x77U;
    weight[36 + index] = 0xFFU;
  }
  for (int32_t row = 16; row < 32; ++row) {
    for (int32_t sub = 0; sub < 4; ++sub) {
      input[row * 36 + sub] = 112U + (row * 4 + sub) % 15U;
    }
    for (int32_t row = 32; row < kRows; ++row) {
      for (int32_t sub = 0; sub < 4; ++sub) {
        input[row * 36 + sub] = 24U + (row * 4 + sub) % 23U;
      }
    }
  }
  for (int32_t column = 2; column < 32; ++column) {
    for (int32_t sub = 0; sub < 4; ++sub) {
      weight[column * 36 + sub] = 112U + (column * 4 + sub) % 15U;
    }
  }
  for (int32_t column = 32; column < 64; ++column) {
    weight[column * 36] = 48U + column % 8U;
    weight[column * 36 + 1] = 112U + column % 15U;
    weight[column * 36 + 2] = 120U + column % 7U;
    weight[column * 36 + 3] = 56U + column % 8U;
  }
  for (int32_t column = 64; column < 96; ++column) {
    for (int32_t sub = 0; sub < 4; ++sub) {
      weight[column * 36 + sub] = 8U + (column * 4 + sub) % 23U;
    }
  }
  for (int32_t sub = 0; sub < 4; ++sub) {
    input[15 * 36 + sub] = 0U;
    weight[256 * 36 + sub] = 0U;
  }
  const auto decode = [](const std::vector<uint8_t> &packed, const int32_t row, const int32_t d) {
    constexpr int32_t kValues[] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const int32_t scale = packed[row * 36 + d / 16];
    const int32_t exponent = scale >> 3;
    const int32_t mantissa = scale & 7;
    const float factor = exponent == 0
                             ? std::ldexp(static_cast<float>(mantissa), -10)
                             : std::ldexp(static_cast<float>(8 + mantissa), exponent - 11);
    const uint8_t codes = packed[row * 36 + 4 + d / 16 * 8 + d % 8];
    return static_cast<float>(kValues[(codes >> (d % 16 < 8 ? 0 : 4)) & 15]) * factor;
  };
  std::vector<float> expected(kRows * kColumns);
  size_t rounding_changes = 0U;
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t column = 0; column < kColumns; ++column) {
      double sum = 0.0;
      for (int32_t d = 0; d < kInner; ++d) {
        sum += static_cast<double>(decode(input, row, d)) * decode(weight, column, d);
      }
      const float nearest = static_cast<float>(sum);
      const float truncated = std::fabs(static_cast<double>(nearest)) > std::fabs(sum)
                                  ? std::nextafter(nearest, 0.0F)
                                  : nearest;
      rounding_changes += truncated != nearest;
      expected[row * kColumns + column] = truncated;
    }
  }
  ASSERT_GT(rounding_changes, 0U);
  for (const auto &device : devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<uint8_t> x;
    test::BackendTensor<uint8_t> w;
    test::BackendTensor<float> s;
    test::BackendTensor<float> y;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(w.Initialize(device, weight));
    ASSERT_TRUE(s.Initialize(device, std::vector<float>(kRows * kInner / 16, 1.0F)));
    ASSERT_TRUE(y.Initialize(device, std::vector<float>(expected.size())));
    ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                backend::QuantizedLinear{.input = x.view(),
                                                         .weight = w.view(),
                                                         .weight_type = 40,
                                                         .weight_scale = 1.0F,
                                                         .rows = kRows,
                                                         .inner = kInner,
                                                         .columns = kColumns,
                                                         .output = y.view(),
                                                         .activation_scales = s.view(),
                                                         .activation_block_size = 16,
                                                         .is_packed_nvfp4 = true,
                                                         .is_ggml_split_k = true}));
    std::vector<float> actual;
    ASSERT_TRUE(y.ReadFloats(&actual));
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t index = 0U; index < actual.size(); ++index) {
      ASSERT_EQ(actual[index], expected[index])
          << "row=" << index / kColumns << " column=" << index % kColumns;
    }
  }
}

TEST_F(QuantizedBackendTest, GgmlSingleTokenQ8PreservesWideWarpReduction) {
  constexpr int32_t kColumns = 17;
  bool has_rounding_change = false;
  for (const int32_t inner : {5120, 6144, 17408}) {
    std::vector<uint16_t> packed;
    std::vector<float> weights;
    ASSERT_TRUE(MakeWeights(8, inner * kColumns, &packed, &weights));
    std::vector<float> codes(inner);
    std::vector<float> scales(inner / 32);
    for (int32_t d = 0; d < inner; ++d) {
      codes[d] = static_cast<float>((d * 37) % 255 - 127);
    }
    for (int32_t block = 0; block < inner / 32; ++block) {
      scales[block] = test::BackendRound<Float16>(static_cast<float>(block * 17 % 31 + 1) * 0.001F);
    }
    const auto reference = [&](const int32_t warps) {
      std::vector<float> results(kColumns);
      for (int32_t column = 0; column < kColumns; ++column) {
        float partials[8][32] = {};
        for (int32_t warp = 0; warp < warps; ++warp) {
          for (int32_t lane = 0; lane < 32; ++lane) {
            for (int32_t first = (warp * 32 + lane) * 8; first < inner; first += warps * 256) {
              int32_t dot = 0;
              for (int32_t j = 0; j < 8; ++j) {
                dot += static_cast<int32_t>(weights[column * inner + first + j] * 32.0F) *
                       static_cast<int32_t>(codes[first + j]);
              }
              partials[warp][lane] = std::fma(static_cast<float>(dot),
                                              0.03125F * scales[first / 32], partials[warp][lane]);
            }
          }
        }
        for (int32_t lane = 0; lane < 32; ++lane) {
          for (int32_t warp = 1; warp < warps; ++warp) {
            partials[0][lane] += partials[warp][lane];
          }
        }
        for (int32_t distance = 16; distance > 0; distance >>= 1) {
          std::array<float, 32U> previous;
          std::copy_n(partials[0], 32, previous.begin());
          for (int32_t lane = 0; lane < 32; ++lane) {
            partials[0][lane] = previous[lane] + previous[lane ^ distance];
          }
        }
        results[column] = partials[0][0] * 0.25F;
      }
      return results;
    };
    const auto narrow = reference(4);
    const auto wide = reference(8);
    has_rounding_change |= narrow != wide;
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      SCOPED_TRACE(inner);
      test::BackendTensor<float> x;
      test::BackendTensor<uint16_t> w;
      test::BackendTensor<float> s;
      test::BackendTensor<float> y;
      ASSERT_TRUE(x.Initialize(device, codes));
      ASSERT_TRUE(w.Initialize(device, packed));
      ASSERT_TRUE(s.Initialize(device, scales));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(kColumns)));
      for (const bool is_ggml : {false, true}) {
        const backend::QuantizedLinear operation{.input = x.view(),
                                                 .weight = w.view(),
                                                 .weight_type = 8,
                                                 .weight_scale = 0.25F,
                                                 .rows = 1,
                                                 .inner = inner,
                                                 .columns = kColumns,
                                                 .output = y.view(),
                                                 .activation_scales = s.view(),
                                                 .activation_block_size = 32,
                                                 .is_ggml_split_k = is_ggml};
        EXPECT_EQ(runtime::GgmlGemvWarps(operation), is_ggml && inner != 5120 ? 8 : 4);
        ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, operation));
        std::vector<float> actual;
        ASSERT_TRUE(y.ReadFloats(&actual));
        EXPECT_EQ(actual, is_ggml && inner != 5120 ? wide : narrow);
      }
    }
  }
  EXPECT_TRUE(has_rounding_change);
}

TEST_F(QuantizedBackendTest, GgmlSplitKPreservesReferencePartitionAndFixupOrder) {
  const std::array<std::array<int32_t, 3U>, 5U> shapes = {std::array<int32_t, 3U>{512, 48, 5120},
                                                          {192, 67, 512},
                                                          {129, 513, 512},
                                                          {10, 48, 5120},
                                                          {129, 3201, 512}};
  bool has_rounding_change = false;
  for (const auto &shape : shapes) {
    const int32_t rows = shape[0];
    const int32_t columns = shape[1];
    const int32_t inner = shape[2];
    const int32_t blocks = inner / 32;
    const auto input = test::BackendInput<float>(static_cast<size_t>(rows) * inner, 5);
    std::vector<uint16_t> storage;
    std::vector<float> weights;
    ASSERT_TRUE(MakeWeights(8, inner * columns, &storage, &weights));
    std::array<std::vector<float>, 2U> outputs;
    for (size_t device_index = 0U; device_index < devices().size(); ++device_index) {
      const auto &device = devices()[device_index];
      SCOPED_TRACE(backend::BackendName(device->kind()));
      SCOPED_TRACE(rows);
      SCOPED_TRACE(columns);
      test::BackendTensor<float> x;
      test::BackendTensor<float> q;
      test::BackendTensor<float> s;
      test::BackendTensor<uint16_t> w;
      test::BackendTensor<float> y;
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(q.Initialize(device, std::vector<float>(input.size())));
      ASSERT_TRUE(s.Initialize(device, std::vector<float>(static_cast<size_t>(rows) * blocks)));
      ASSERT_TRUE(w.Initialize(device, storage));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(static_cast<size_t>(rows) * columns)));
      const backend::QuantizeActivation quantize{.input = x.view(),
                                                 .weight_type = 8,
                                                 .rows = rows,
                                                 .columns = inner,
                                                 .is_decode = false,
                                                 .output = q.view(),
                                                 .scales = s.view(),
                                                 .is_ggml_rounding = true};
      ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, quantize));
      std::vector<float> codes;
      std::vector<float> scales;
      ASSERT_TRUE(q.ReadFloats(&codes));
      ASSERT_TRUE(s.ReadFloats(&scales));
      const int32_t groups = runtime::CudaKernelRuntime::MultiprocessorCount();
      const int32_t token_tile = runtime::GgmlTokenTile(rows);
      const int64_t token_tiles = (rows - 1) / token_tile + 1;
      const int64_t tile_count = ((columns - 1) / 128 + 1) * token_tiles;
      const int64_t waves = (tile_count - 1) / groups + 1;
      const int32_t partitions = 100 * tile_count / (groups * waves) >= 90
                                     ? static_cast<int32_t>(tile_count)
                                     : static_cast<int32_t>(std::min<int64_t>(
                                           groups, tile_count * ((inner - 1) / 256 + 1)));
      std::vector<float> expected(static_cast<size_t>(rows) * columns);
      for (int32_t row = 0; row < rows; ++row) {
        for (int32_t column = 0; column < columns; ++column) {
          std::vector<float> weighted(blocks);
          for (int32_t block = 0; block < blocks; ++block) {
            int32_t dot = 0;
            for (int32_t d = 0; d < 32; ++d) {
              dot +=
                  static_cast<int32_t>(codes[static_cast<size_t>(row) * inner + block * 32 + d]) *
                  static_cast<int32_t>(
                      weights[static_cast<size_t>(column) * inner + block * 32 + d] * 32.0F);
            }
            weighted[block] = static_cast<float>(dot) * 0.03125F;
          }
          float sequential = 0.0F;
          for (int32_t block = 0; block < blocks; ++block) {
            sequential = std::fma(weighted[block],
                                  scales[static_cast<size_t>(row) * blocks + block], sequential);
          }
          const int64_t tile = column / 128 * token_tiles + row / token_tile;
          float last = 0.0F;
          float previous = 0.0F;
          bool has_last = false;
          for (int32_t part = partitions - 1; part >= 0; --part) {
            int64_t first = static_cast<int64_t>(part) * tile_count * blocks / partitions;
            int64_t stop = static_cast<int64_t>(part + 1) * tile_count * blocks / partitions;
            first -= first % blocks % 8;
            stop -= stop % blocks % 8;
            first = std::max(first, tile * blocks) - tile * blocks;
            stop = std::min(stop, (tile + 1) * blocks) - tile * blocks;
            if (first >= stop) {
              continue;
            }
            float partial = 0.0F;
            for (int64_t block = first; block < stop; ++block) {
              partial = std::fma(weighted[block], scales[static_cast<size_t>(row) * blocks + block],
                                 partial);
            }
            if (!has_last) {
              last = partial;
              has_last = true;
            } else {
              previous += partial;
            }
          }
          expected[static_cast<size_t>(row) * columns + column] = last + previous;
          has_rounding_change = has_rounding_change || last + previous != sequential;
        }
      }
      const backend::QuantizedLinear operation{.input = q.view(),
                                               .weight = w.view(),
                                               .weight_type = 8,
                                               .weight_scale = 1.0F,
                                               .rows = rows,
                                               .inner = inner,
                                               .columns = columns,
                                               .output = y.view(),
                                               .activation_scales = s.view(),
                                               .activation_block_size = 32,
                                               .is_ggml_split_k = true};
      ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, operation));
      ASSERT_TRUE(y.ReadFloats(&outputs[device_index]));
      EXPECT_EQ(outputs[device_index], expected);
      const auto view = y.view();
      const backend::Allocation readback{.data = view.data,
                                         .size_bytes = view.size_bytes,
                                         .memory_kind = backend::MemoryKind::kDeviceView,
                                         .offset_bytes = view.offset_bytes};
      std::unique_ptr<backend::ExecutionPlan> plan;
      ASSERT_TRUE(device->CreateExecutionPlan(
          backend::ScalarType::kFloat32,
          std::vector<backend::ComputeOperation>{quantize, operation}, readback, &plan));
      double elapsed_ms = 0.0;
      ASSERT_TRUE(plan->Run(view.size_bytes, outputs[device_index].data(), &elapsed_ms));
      EXPECT_EQ(outputs[device_index], expected);
    }
    EXPECT_EQ(outputs[0], outputs[1]);
  }
  EXPECT_TRUE(has_rounding_change);
}

TEST_F(QuantizedBackendTest, PackedNvfp4ActivationsPreserveValuesAndCapturedProducts) {
  const std::array<std::array<int32_t, 3U>, 3U> shapes = {
      std::array<int32_t, 3U>{64, 3, 1}, {192, 67, 9}, {5120, 131, 129}};
  constexpr int32_t kFp4Values[] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  for (const auto &shape : shapes) {
    const int32_t inner = shape[0];
    const int32_t columns = shape[1];
    const int32_t rows = shape[2];
    const size_t count = static_cast<size_t>(rows) * inner;
    auto input = test::BackendInput<float>(count, 3);
    std::fill_n(input.begin(), inner, 0.0F);
    std::vector<uint16_t> weight_storage;
    std::vector<float> weights;
    ASSERT_TRUE(MakeWeights(40, inner * columns, &weight_storage, &weights));
    std::array<std::vector<uint16_t>, 2U> packed_results;
    for (size_t device_index = 0U; device_index < devices().size(); ++device_index) {
      const auto &device = devices()[device_index];
      SCOPED_TRACE(backend::BackendName(device->kind()));
      SCOPED_TRACE(inner);
      SCOPED_TRACE(rows);
      test::BackendTensor<float> x;
      test::BackendTensor<float> q;
      test::BackendTensor<float> scales;
      test::BackendTensor<float> packed_scales;
      test::BackendTensor<uint16_t> packed;
      test::BackendTensor<uint16_t> w;
      test::BackendTensor<float> y;
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(q.Initialize(device, std::vector<float>(count)));
      ASSERT_TRUE(scales.Initialize(device, std::vector<float>(count / 16U)));
      ASSERT_TRUE(packed_scales.Initialize(device, std::vector<float>(count / 16U)));
      ASSERT_TRUE(packed.Initialize(device, std::vector<uint16_t>(count / 64U * 18U)));
      ASSERT_TRUE(w.Initialize(device, weight_storage));
      ASSERT_TRUE(y.Initialize(device, std::vector<float>(static_cast<size_t>(rows) * columns)));
      backend::QuantizeActivation quantize{.input = x.view(),
                                           .weight_type = 40,
                                           .rows = rows,
                                           .columns = inner,
                                           .is_decode = false,
                                           .output = q.view(),
                                           .scales = scales.view(),
                                           .is_ggml_rounding = true};
      ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, quantize));
      std::vector<float> codes;
      std::vector<float> factors;
      ASSERT_TRUE(q.ReadFloats(&codes));
      ASSERT_TRUE(scales.ReadFloats(&factors));
      quantize.output = packed.view();
      quantize.scales = packed_scales.view();
      quantize.is_packed_nvfp4 = true;
      ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, quantize));
      ASSERT_TRUE(packed.Read(&packed_results[device_index]));
      std::vector<float> packed_factors;
      ASSERT_TRUE(packed_scales.ReadFloats(&packed_factors));
      EXPECT_EQ(factors, packed_factors);
      const uint8_t *const bytes =
          reinterpret_cast<const uint8_t *>(packed_results[device_index].data());
      std::vector<float> decoded(count);
      for (size_t index = 0U; index < count; ++index) {
        const uint8_t *const block = bytes + index / 64U * 36U;
        const uint8_t scale = block[index % 64U / 16U];
        const int32_t exponent = scale >> 3U;
        const int32_t mantissa = scale & 7U;
        const float factor = exponent == 0 ? std::ldexp(float(mantissa), -10)
                                           : std::ldexp(float(8 + mantissa), exponent - 11);
        const uint8_t pair = block[4U + index % 64U / 16U * 8U + index % 8U];
        const uint8_t code = (pair >> (index % 16U < 8U ? 0U : 4U)) & 15U;
        decoded[index] = kFp4Values[code] * factor;
      }
      EXPECT_EQ(codes, decoded);
      std::vector<float> expected(static_cast<size_t>(rows) * columns);
      for (int32_t row = 0; row < rows; ++row) {
        for (int32_t column = 0; column < columns; ++column) {
          double sum = 0.0;
          for (int32_t d = 0; d < inner; ++d) {
            sum += static_cast<double>(codes[static_cast<size_t>(row) * inner + d]) *
                   weights[static_cast<size_t>(column) * inner + d];
          }
          expected[static_cast<size_t>(row) * columns + column] =
              (static_cast<float>(sum) * factors[static_cast<size_t>(row) * (inner / 16)]) * 0.25F;
        }
      }
      const backend::QuantizedLinear operation{.input = packed.view(),
                                               .weight = w.view(),
                                               .weight_type = 40,
                                               .weight_scale = 0.25F,
                                               .rows = rows,
                                               .inner = inner,
                                               .columns = columns,
                                               .output = y.view(),
                                               .activation_scales = packed_scales.view(),
                                               .activation_block_size = 16,
                                               .is_packed_nvfp4 = true};
      ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, operation));
      std::vector<float> actual;
      ASSERT_TRUE(y.ReadFloats(&actual));
      test::ExpectBackendClose(actual, expected, {5.0e-5F, 2.0e-5F});
      const auto view = y.view();
      const backend::Allocation readback{.data = view.data,
                                         .size_bytes = view.size_bytes,
                                         .memory_kind = backend::MemoryKind::kDeviceView,
                                         .offset_bytes = view.offset_bytes};
      std::unique_ptr<backend::ExecutionPlan> plan;
      ASSERT_TRUE(device->CreateExecutionPlan(
          backend::ScalarType::kFloat32,
          std::vector<backend::ComputeOperation>{quantize, operation}, readback, &plan));
      double elapsed_ms = 0.0;
      ASSERT_TRUE(plan->Run(view.size_bytes, actual.data(), &elapsed_ms));
      EXPECT_GT(elapsed_ms, 0.0);
      test::ExpectBackendClose(actual, expected, {5.0e-5F, 2.0e-5F});
    }
    EXPECT_EQ(packed_results[0], packed_results[1]);
  }
}

TEST_F(QuantizedBackendTest, PackedNvfp4PreservesK64RoundingAndScaledSplitKFixup) {
  const std::array<std::array<int32_t, 3U>, 4U> shapes = {
      std::array<int32_t, 3U>{9, 67, 5120}, {192, 3, 512}, {129, 131, 1024}, {17, 8193, 1024}};
  bool has_split_rounding_change = false;
  for (const auto &shape : shapes) {
    const int32_t rows = shape[0];
    const int32_t columns = shape[1];
    const int32_t inner = shape[2];
    const int32_t blocks = inner / 64;
    std::vector<uint16_t> input_storage;
    std::vector<uint16_t> weight_storage;
    std::vector<float> input;
    std::vector<float> weights;
    ASSERT_TRUE(MakeWeights(40, rows * inner, &input_storage, &input));
    ASSERT_TRUE(MakeWeights(40, columns * inner, &weight_storage, &weights));
    // Uniform scales within each K64 make its exact dot representable in FP32.
    for (const bool is_input : {true, false}) {
      auto &storage = is_input ? input_storage : weight_storage;
      auto &values = is_input ? input : weights;
      uint8_t *const bytes = reinterpret_cast<uint8_t *>(storage.data());
      for (size_t block = 0U; block < values.size() / 64U; ++block) {
        const uint8_t code = static_cast<uint8_t>(40U + block % 9U * 8U);
        const float factor = std::ldexp(1.0F, static_cast<int32_t>(code / 8U) - 8);
        for (size_t sub = 0U; sub < 4U; ++sub) {
          const float old_factor =
              sub == 0U ? std::ldexp(1.0F, -10) : std::ldexp(1.0F, static_cast<int32_t>(sub) - 3);
          bytes[block * 36U + sub] = code;
          for (size_t d = 0U; d < 16U; ++d) {
            values[block * 64U + sub * 16U + d] *= factor / old_factor;
          }
        }
      }
    }
    std::vector<float> scales(static_cast<size_t>(rows) * (inner / 16));
    for (int32_t row = 0; row < rows; ++row) {
      std::fill_n(scales.begin() + static_cast<size_t>(row) * (inner / 16), inner / 16,
                  static_cast<float>(row + 1) * 0.001273F);
    }
    std::vector<float> dots(static_cast<size_t>(rows) * columns * blocks);
    for (int32_t row = 0; row < rows; ++row) {
      for (int32_t column = 0; column < columns; ++column) {
        for (int32_t block = 0; block < blocks; ++block) {
          double dot = 0.0;
          for (int32_t d = 0; d < 64; ++d) {
            dot += static_cast<double>(input[static_cast<size_t>(row) * inner + block * 64 + d]) *
                   weights[static_cast<size_t>(column) * inner + block * 64 + d];
          }
          dots[(static_cast<size_t>(row) * columns + column) * blocks + block] =
              static_cast<float>(dot);
        }
      }
    }
    std::vector<float> sequential(static_cast<size_t>(rows) * columns);
    for (const bool is_split_k : {false, true}) {
      const int32_t token_tile = runtime::GgmlTokenTile(rows);
      const int64_t token_tiles = (rows - 1) / token_tile + 1;
      const int64_t tiles = ((columns - 1) / 128 + 1) * token_tiles;
      const int32_t compute_units = runtime::CudaKernelRuntime::MultiprocessorCount();
      const int64_t waves = (tiles - 1) / compute_units + 1;
      const int32_t parts = !is_split_k ? 1
                            : 100 * tiles / (compute_units * waves) >= 90
                                ? static_cast<int32_t>(tiles)
                                : static_cast<int32_t>(std::min<int64_t>(
                                      compute_units, tiles * ((inner - 1) / 512 + 1)));
      std::vector<float> expected(sequential.size());
      for (int32_t row = 0; row < rows; ++row) {
        for (int32_t column = 0; column < columns; ++column) {
          const size_t index = static_cast<size_t>(row) * columns + column;
          const int64_t tile = column / 128 * token_tiles + row / token_tile;
          float last = 0.0F;
          float previous = 0.0F;
          bool has_last = false;
          for (int32_t part = parts - 1; part >= 0; --part) {
            int64_t first = 0;
            int64_t stop = blocks;
            if (is_split_k) {
              first = static_cast<int64_t>(part) * tiles * blocks / parts;
              stop = static_cast<int64_t>(part + 1) * tiles * blocks / parts;
              first -= first % blocks % 8;
              stop -= stop % blocks % 8;
              first = std::max(first, tile * blocks) - tile * blocks;
              stop = std::min(stop, (tile + 1) * blocks) - tile * blocks;
            }
            if (first >= stop) {
              continue;
            }
            float partial = 0.0F;
            for (int64_t block = first; block < stop; ++block) {
              partial += dots[index * blocks + block];
            }
            partial *= scales[static_cast<size_t>(row) * (inner / 16)];
            if (!has_last) {
              last = partial;
              has_last = true;
            } else {
              previous += partial;
            }
          }
          expected[index] = (last + previous) * 0.25F;
          has_split_rounding_change =
              has_split_rounding_change || (is_split_k && expected[index] != sequential[index]);
        }
      }
      if (!is_split_k) {
        sequential = expected;
      }
      for (const auto &device : devices()) {
        SCOPED_TRACE(backend::BackendName(device->kind()));
        SCOPED_TRACE(rows);
        SCOPED_TRACE(inner);
        SCOPED_TRACE(is_split_k);
        test::BackendTensor<uint16_t> x;
        test::BackendTensor<uint16_t> w;
        test::BackendTensor<float> s;
        test::BackendTensor<float> y;
        ASSERT_TRUE(x.Initialize(device, input_storage));
        ASSERT_TRUE(w.Initialize(device, weight_storage));
        ASSERT_TRUE(s.Initialize(device, scales));
        ASSERT_TRUE(y.Initialize(device, std::vector<float>(expected.size())));
        const backend::QuantizedLinear operation{.input = x.view(),
                                                 .weight = w.view(),
                                                 .weight_type = 40,
                                                 .weight_scale = 0.25F,
                                                 .rows = rows,
                                                 .inner = inner,
                                                 .columns = columns,
                                                 .output = y.view(),
                                                 .activation_scales = s.view(),
                                                 .activation_block_size = 16,
                                                 .is_packed_nvfp4 = true,
                                                 .is_ggml_split_k = is_split_k};
        ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, operation));
        std::vector<float> actual;
        ASSERT_TRUE(y.ReadFloats(&actual));
        EXPECT_EQ(actual, expected);
        const auto view = y.view();
        const backend::Allocation readback{.data = view.data,
                                           .size_bytes = view.size_bytes,
                                           .memory_kind = backend::MemoryKind::kDeviceView,
                                           .offset_bytes = view.offset_bytes};
        std::unique_ptr<backend::ExecutionPlan> plan;
        ASSERT_TRUE(device->CreateExecutionPlan(backend::ScalarType::kFloat32, {operation},
                                                readback, &plan));
        double elapsed_ms = 0.0;
        ASSERT_TRUE(plan->Run(view.size_bytes, actual.data(), &elapsed_ms));
        EXPECT_EQ(actual, expected);
      }
    }
  }
  EXPECT_TRUE(has_split_rounding_change);
}

TEST(QuantizedLayoutTest, TwoPartBoundPreservesAlignedPartitionCoverage) {
  int64_t verified_tiles = 0;
  for (const int32_t inner : {192, 512, 1024, 5120, 17408}) {
    for (const int32_t rows : {17, 129, 512}) {
      for (const int32_t columns : {129, 2049, 8193}) {
        const backend::QuantizedLinear args{
            .weight_type = 40, .rows = rows, .inner = inner, .columns = columns};
        for (const int32_t compute_units : {1, 3, 7, 48, 80}) {
          if (!runtime::HasAtMostTwoGgmlParts(args, compute_units)) {
            continue;
          }
          const int64_t blocks = inner / 64;
          const int64_t token_tiles = (rows - 1) / runtime::GgmlTokenTile(rows) + 1;
          const int64_t tiles = ((columns - 1) / 128 + 1) * token_tiles;
          const int32_t parts = runtime::GgmlStreamGroups(args, compute_units);
          for (int64_t tile = 0; tile < tiles; ++tile) {
            int32_t nonempty_parts = 0;
            int64_t covered_blocks = 0;
            for (int32_t part = 0; part < parts; ++part) {
              int64_t first = static_cast<int64_t>(part) * tiles * blocks / parts;
              int64_t stop = static_cast<int64_t>(part + 1) * tiles * blocks / parts;
              first -= first % blocks % 8;
              stop -= stop % blocks % 8;
              const int64_t begin = std::max(first, tile * blocks);
              const int64_t end = std::min(stop, (tile + 1) * blocks);
              if (begin < end) {
                ++nonempty_parts;
                covered_blocks += end - begin;
              }
            }
            ASSERT_GE(nonempty_parts, 1);
            ASSERT_LE(nonempty_parts, 2);
            ASSERT_EQ(covered_blocks, blocks);
            ++verified_tiles;
          }
          EXPECT_GT(verified_tiles, 0);
        }
      }
    }
  }
}

TEST_F(QuantizedBackendTest, PackedProductsHandleLargeInnerDimensionsAndCapturedReplay) {
  const std::array<std::array<int32_t, 3U>, 3U> shapes = {
      std::array<int32_t, 3U>{64, 9, 8}, {5120, 67, 65}, {32832, 9, 1}};
  for (const auto &shape : shapes) {
    const int32_t inner = shape[0];
    const int32_t columns = shape[1];
    const int32_t rows = shape[2];
    for (const auto &format : {std::pair{8, 32}, std::pair{40, 32}, std::pair{40, 16}}) {
      const int32_t type = format.first;
      const int32_t block = format.second;
      SCOPED_TRACE(inner);
      SCOPED_TRACE(type);
      SCOPED_TRACE(block);
      std::vector<uint16_t> packed;
      std::vector<float> weights;
      ASSERT_TRUE(MakeWeights(type, inner * columns, &packed, &weights));
      std::vector<float> input(static_cast<size_t>(rows) * inner);
      std::vector<float> scales(static_cast<size_t>(rows) * (inner / block));
      for (size_t index = 0U; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int32_t>(index % 15U) - 7);
      }
      for (int32_t row = 0; row < rows; ++row) {
        for (int32_t first = 0; first < inner; first += block) {
          scales[static_cast<size_t>(row) * (inner / block) + first / block] =
              std::ldexp(1.0F, -10 + (block == 16 ? row % 3 : (row + first / block) % 3));
        }
      }
      std::vector<float> expected(static_cast<size_t>(rows) * columns);
      for (int32_t row = 0; row < rows; ++row) {
        for (int32_t column = 0; column < columns; ++column) {
          double sum = 0.0;
          for (int32_t d = 0; d < inner; ++d) {
            const size_t index = static_cast<size_t>(row) * inner + d;
            sum += input[index] * scales[index / block] *
                   weights[static_cast<size_t>(column) * inner + d];
          }
          expected[static_cast<size_t>(row) * columns + column] = static_cast<float>(sum) * 0.25F;
        }
      }
      for (const auto &device : devices()) {
        SCOPED_TRACE(backend::BackendName(device->kind()));
        test::BackendTensor<float> x;
        test::BackendTensor<float> s;
        test::BackendTensor<uint16_t> w;
        test::BackendTensor<float> y;
        ASSERT_TRUE(x.Initialize(device, input));
        ASSERT_TRUE(s.Initialize(device, scales));
        ASSERT_TRUE(w.Initialize(device, packed));
        ASSERT_TRUE(y.Initialize(device, std::vector<float>(expected.size())));
        const backend::QuantizedLinear operation{x.view(), w.view(), type,     0.25F,    rows,
                                                 inner,    columns,  y.view(), s.view(), block};
        ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, operation));
        std::vector<float> actual;
        ASSERT_TRUE(y.ReadFloats(&actual));
        test::ExpectBackendClose(actual, expected, {1.0e-5F, 1.0e-6F});
        if (rows != 65) {
          continue;
        }
        const auto view = y.view();
        const backend::Allocation readback{.data = view.data,
                                           .size_bytes = view.size_bytes,
                                           .memory_kind = backend::MemoryKind::kDeviceView,
                                           .offset_bytes = view.offset_bytes};
        std::unique_ptr<backend::ExecutionPlan> plan;
        ASSERT_TRUE(device->CreateExecutionPlan(backend::ScalarType::kFloat32,
                                                std::vector<backend::ComputeOperation>{operation},
                                                readback, &plan));
        for (int32_t replay = 0; replay < 2; ++replay) {
          double elapsed_ms = 0.0;
          ASSERT_TRUE(plan->Run(view.size_bytes, actual.data(), &elapsed_ms));
          EXPECT_GT(elapsed_ms, 0.0);
          test::ExpectBackendClose(actual, expected, {1.0e-5F, 1.0e-6F});
        }
      }
    }
  }
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
