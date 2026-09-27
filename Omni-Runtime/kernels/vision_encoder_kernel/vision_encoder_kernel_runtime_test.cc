#include <cstdint>
#include <cmath>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/vision_encoder_kernel/vision_encoder_cuda_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kFrames = 2;
constexpr int32_t kHeight = 16;
constexpr int32_t kWidth = 16;
constexpr int32_t kChannels = 3;
constexpr int32_t kOutputChannels = 1152;
constexpr int32_t kTemporalPatch = 2;
constexpr int32_t kSpatialPatch = 16;
constexpr int32_t kPatchVolume = kChannels * kTemporalPatch * kSpatialPatch * kSpatialPatch;
constexpr int32_t kPatchCount = 1;

class VisionEncoderKernelTest : public test::KernelTestFixture {};

template <typename T> class VisionPatchLowPrecisionTest : public test::KernelTestFixture {};
using PatchTypes = ::testing::Types<Float16, BFloat16>;
TYPED_TEST_SUITE(VisionPatchLowPrecisionTest, PatchTypes);

TYPED_TEST(VisionPatchLowPrecisionTest, AccumulatesBothFramesBeforeRoundingWithTileTails) {
  constexpr int32_t kPatches = 65;
  constexpr int32_t kColumns = 67;
  for (const int32_t inner : {35, 768}) {
    SCOPED_TRACE(inner);
    std::vector<TypeParam> input(kPatches * inner);
    std::vector<TypeParam> first_weight(kColumns * inner);
    std::vector<TypeParam> second_weight(kColumns * inner);
    std::vector<TypeParam> bias(kColumns);
    for (size_t index = 0U; index < input.size(); ++index) {
      input[index] = test::LowPrecision<TypeParam>((static_cast<int32_t>(index % 7U) - 3) / 32.0F);
    }
    for (size_t index = 0U; index < first_weight.size(); ++index) {
      first_weight[index] =
          test::LowPrecision<TypeParam>((static_cast<int32_t>(index % 11U) - 5) / 16.0F);
      second_weight[index] =
          test::LowPrecision<TypeParam>((static_cast<int32_t>(index % 13U) - 6) / 8.0F);
    }
    for (int32_t index = 0; index < kColumns; ++index) {
      bias[index] = test::LowPrecision<TypeParam>((index % 5 - 2) / 16.0F);
    }
    const auto x = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), input);
    const auto w0 = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), first_weight);
    const auto w1 = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), second_weight);
    const auto b = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), bias);
    const auto y =
        memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), kPatches * kColumns);
    const int32_t patch = inner == 768 ? 16 : 1;
    ASSERT_TRUE((runtime::VisionPatchEmbedPairKernelRuntime<TypeParam>::CompileAndLaunch(
        "vision_patch_pair_low_precision", {.options = {},
                                            .input = x->device_address(),
                                            .first_weight = w0->device_address(),
                                            .second_weight = w1->device_address(),
                                            .bias = b->device_address(),
                                            .height = 5 * patch,
                                            .width = 13 * patch,
                                            .channels = inner / (patch * patch),
                                            .output_channels = kColumns,
                                            .spatial_patch = patch,
                                            .output = y->device_address()})));
    std::vector<TypeParam> actual(kPatches * kColumns);
    ASSERT_TRUE(y->CopyToHost(actual.size() * sizeof(TypeParam), actual.data()));
    for (int32_t row = 0; row < kPatches; ++row) {
      for (int32_t column = 0; column < kColumns; ++column) {
        float sum = test::HostFloat(bias[column]);
        for (int32_t d = 0; d < inner; ++d) {
          sum += test::HostFloat(input[row * inner + d]) *
                 (test::HostFloat(first_weight[column * inner + d]) +
                  test::HostFloat(second_weight[column * inner + d]));
        }
        EXPECT_FLOAT_EQ(test::HostFloat(actual[row * kColumns + column]),
                        test::HostFloat(test::LowPrecision<TypeParam>(sum)));
      }
    }
  }
}

TEST_F(VisionEncoderKernelTest, DistinguishesBlockAndMergerActivations) {
  const std::vector<float> input = {-5.0F, -3.0F, -1.0F, 0.0F, 1.0F, 2.0F, 3.0F, 5.0F};
  const std::vector<float> bias = {0.125F, -0.25F, 0.5F, 0.0F};
  const auto input_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto bias_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), bias);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), input.size());
  for (const bool is_approximate : {false, true}) {
    ASSERT_TRUE((runtime::VisionGeluBiasKernelRuntime<float>::CompileAndLaunch(
        "vision_gelu", {.options = {.grid_dim = {1U, 1U, 1U},
                                    .block_dim = {256U, 1U, 1U},
                                    .shared_memory_bytes = 0},
                        .input = input_device->device_address(),
                        .bias = bias_device->device_address(),
                        .element_count = input.size(),
                        .row_size = static_cast<int32_t>(bias.size()),
                        .is_approximate = is_approximate,
                        .output = output_device->device_address()})));
    std::vector<float> actual(input.size());
    ASSERT_TRUE(output_device->CopyToHost(actual.size() * sizeof(float), actual.data()));
    for (size_t index = 0U; index < input.size(); ++index) {
      const float value = input[index] + bias[index % bias.size()];
      const float activation =
          is_approximate
              ? std::tanh(0.7978845608028654F * (value + 0.044715F * value * value * value))
              : std::erf(value * 0.7071067811865475F);
      EXPECT_NEAR(actual[index], 0.5F * value * (1.0F + activation), 2.0e-6F);
    }
  }
}

TEST_F(VisionEncoderKernelTest, EmbedsQwen3VlVisionPatch) {
  std::vector<float> input(static_cast<std::size_t>(kFrames) * kHeight * kWidth * kChannels);
  std::vector<float> weight(static_cast<std::size_t>(kOutputChannels) * kPatchVolume);
  std::vector<float> bias(kOutputChannels);
  std::vector<float> expected(kOutputChannels);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = 0.001f * static_cast<float>(index) - 0.768f;
  }
  for (int32_t output_channel = 0; output_channel < kOutputChannels; ++output_channel) {
    bias[output_channel] = 0.0001f * static_cast<float>(output_channel);
    float value = bias[output_channel];
    for (int32_t patch_index = 0; patch_index < kPatchVolume; ++patch_index) {
      const float weight_value = 0.000001f * static_cast<float>(patch_index) +
                                 0.0000001f * static_cast<float>(output_channel);
      weight[static_cast<int64_t>(output_channel) * kPatchVolume + patch_index] = weight_value;
      value += input[patch_index] * weight_value;
    }
    expected[output_channel] = value;
  }
  const auto input_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), input);
  const auto weight_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), weight);
  const auto bias_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), bias);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE((runtime::VisionPatchEmbedKernelRuntime<float>::CompileAndLaunch(
      "vision_patch_embed_float",
      {.options = {.grid_dim = dim3(kPatchCount * kOutputChannels, 1, 1),
                   .block_dim = dim3(128, 1, 1),
                   .shared_memory_bytes = 0},
       .input = input_device->device_address(),
       .weight = weight_device->device_address(),
       .bias = bias_device->device_address(),
       .frames = kFrames,
       .height = kHeight,
       .width = kWidth,
       .channels = kChannels,
       .output_channels = kOutputChannels,
       .temporal_patch = kTemporalPatch,
       .spatial_patch = kSpatialPatch,
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

template <typename T> class VisionBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(VisionBackendTest, test::BackendTypes);

class MixedVisionBackendTest : public test::BackendParityFixture {};

TEST_F(MixedVisionBackendTest, GgmlGeluMatchesReferenceAndAcrossBackends) {
  constexpr int32_t kRow = 31;
  const auto input = test::BackendInput<float>(1027, 4);
  const auto bias = test::BackendInput<float>(kRow, 7);
  std::vector<float> expected(input.size());
  for (size_t index = 0U; index < input.size(); ++index) {
    const float x = input[index] + bias[index % kRow];
    expected[index] =
        0.5F * x * (1.0F + std::tanh(0.7978845608028654F * x * (1.0F + 0.044715F * x * x)));
  }
  std::array<std::vector<float>, 2U> results;
  for (size_t index = 0U; index < devices().size(); ++index) {
    const auto &device = devices()[index];
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<float> x;
    test::BackendTensor<float> b;
    test::BackendTensor<float> y;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(b.Initialize(device, bias));
    ASSERT_TRUE(y.Initialize(device, std::vector<float>(input.size())));
    backend::GeluBias args{.input = x.view(),
                           .bias = b.view(),
                           .element_count = input.size(),
                           .row_size = kRow,
                           .is_approximate = true,
                           .output = y.view(),
                           .is_ggml_gelu = true};
    ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32, args));
    ASSERT_TRUE(y.ReadFloats(&results[index]));
    test::ExpectBackendClose(results[index], expected, {2.0e-5F, 2.0e-5F});
    args.is_approximate = false;
    EXPECT_FALSE(device->Compute(backend::ScalarType::kFloat32, args));
  }
  EXPECT_EQ(results[0], results[1]);
}

TEST_F(MixedVisionBackendTest, GgmlRotaryMatchesAcrossBackendsWithFloatAndHalfKeyValues) {
  constexpr int32_t kTokens = 86 * 128;
  constexpr int32_t kHidden = 144;
  const auto input = test::BackendInput<float>(kTokens * kHidden * 3U, 9);
  for (const bool is_packed : {false, true}) {
    std::array<std::vector<float>, 2U> results;
    for (size_t index = 0U; index < devices().size(); ++index) {
      const auto &device = devices()[index];
      SCOPED_TRACE(backend::BackendName(device->kind()));
      test::BackendTensor<float> x;
      test::BackendTensor<float> query;
      test::BackendTensor<float> key;
      test::BackendTensor<float> value;
      test::BackendTensor<Float16> half_key;
      test::BackendTensor<Float16> half_value;
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(query.Initialize(device, std::vector<float>(kTokens * kHidden)));
      ASSERT_TRUE(key.Initialize(device, std::vector<float>(kTokens * kHidden)));
      ASSERT_TRUE(value.Initialize(device, std::vector<float>(kTokens * kHidden)));
      ASSERT_TRUE(half_key.Initialize(device, std::vector<Float16>(kTokens * kHidden)));
      ASSERT_TRUE(half_value.Initialize(device, std::vector<Float16>(kTokens * kHidden)));
      ASSERT_TRUE(device->Compute(
          backend::ScalarType::kFloat32,
          backend::GridQkvRope{.qkv = x.view(),
                               .token_count = kTokens,
                               .grid_height = 86,
                               .grid_width = 128,
                               .merge_size = 2,
                               .head_count = 2,
                               .head_size = 72,
                               .query = query.view(),
                               .key = is_packed ? half_key.view() : key.view(),
                               .value = is_packed ? half_value.view() : value.view(),
                               .is_fp16_key_value = is_packed,
                               .is_ggml_rope = true}));
      ASSERT_TRUE(query.ReadFloats(&results[index]));
      std::vector<float> extra;
      ASSERT_TRUE(is_packed ? half_key.ReadFloats(&extra) : key.ReadFloats(&extra));
      results[index].insert(results[index].end(), extra.begin(), extra.end());
      ASSERT_TRUE(is_packed ? half_value.ReadFloats(&extra) : value.ReadFloats(&extra));
      results[index].insert(results[index].end(), extra.begin(), extra.end());
    }
    EXPECT_EQ(results[0], results[1]);
  }
}

TEST_F(MixedVisionBackendTest, GgmlPositionInterpolationHandlesNonSquareAndSingletonGrids) {
  constexpr int32_t kHidden = 3;
  const std::array<std::array<int32_t, 4U>, 3U> shapes = {
      {{86, 128, 48, 2}, {1, 5, 3, 1}, {4, 6, 1, 2}}};
  for (const auto &[height, width, table_size, merge] : shapes) {
    const auto weight = test::BackendInput<float>(table_size * table_size * kHidden, 6);
    std::vector<float> expected(height * width * kHidden);
    for (int32_t token = 0; token < height * width; ++token) {
      const int32_t group = token / (merge * merge);
      const int32_t local = token % (merge * merge);
      const int32_t row = group / (width / merge) * merge + local / merge;
      const int32_t column = group % (width / merge) * merge + local % merge;
      const double sy = height > 1 ? double(row) * (table_size - 1) / (height - 1) : 0.0;
      const double sx = width > 1 ? double(column) * (table_size - 1) / (width - 1) : 0.0;
      const int32_t y0 = static_cast<int32_t>(sy);
      const int32_t x0 = static_cast<int32_t>(sx);
      const int32_t y1 = std::min(y0 + 1, table_size - 1);
      const int32_t x1 = std::min(x0 + 1, table_size - 1);
      const double dy = sy - y0;
      const double dx = sx - x0;
      for (int32_t d = 0; d < kHidden; ++d) {
        expected[token * kHidden + d] = static_cast<float>(
            weight[(y0 * table_size + x0) * kHidden + d] * (1.0 - dx) * (1.0 - dy) +
            weight[(y0 * table_size + x1) * kHidden + d] * dx * (1.0 - dy) +
            weight[(y1 * table_size + x0) * kHidden + d] * (1.0 - dx) * dy +
            weight[(y1 * table_size + x1) * kHidden + d] * dx * dy);
      }
    }
    for (const auto &device : devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      test::BackendTensor<float> input;
      test::BackendTensor<float> positions;
      test::BackendTensor<float> output;
      ASSERT_TRUE(input.Initialize(device, std::vector<float>(expected.size())));
      ASSERT_TRUE(positions.Initialize(device, weight));
      ASSERT_TRUE(output.Initialize(device, std::vector<float>(expected.size())));
      ASSERT_TRUE(
          device->Compute(backend::ScalarType::kFloat32,
                          backend::GridPositionEmbedding{.input = input.view(),
                                                         .position_weight = positions.view(),
                                                         .grid_height = height,
                                                         .grid_width = width,
                                                         .position_grid_size = table_size,
                                                         .hidden_size = kHidden,
                                                         .merge_size = merge,
                                                         .output = output.view(),
                                                         .is_ggml_interpolation = true}));
      std::vector<float> actual;
      ASSERT_TRUE(output.ReadFloats(&actual));
      test::ExpectBackendClose(actual, expected, {2.0e-5F, 2.0e-6F});
    }
  }
}

TEST_F(MixedVisionBackendTest, PairedProjectionRoundsEachFrameBeforeAddingBias) {
  constexpr int32_t kPatches = 15;
  constexpr int32_t kInner = 12;
  constexpr int32_t kColumns = 7;
  auto input = test::BackendInput<float>(kPatches * kInner, 1);
  const auto first = test::BackendInput<float>(kColumns * kInner, 3);
  const auto second = test::BackendInput<float>(first.size(), 5);
  const auto bias = test::BackendInput<float>(kColumns, 7);
  for (float &value : input) {
    value *= 0.12123F;
  }
  const auto round = [](const float value) { return test::BackendRound<Float16>(value); };
  std::vector<float> expected(kPatches * kColumns);
  for (int32_t row = 0; row < kPatches; ++row) {
    for (int32_t column = 0; column < kColumns; ++column) {
      float a = 0.0F;
      float b = 0.0F;
      for (int32_t d = 0; d < kInner; ++d) {
        const float x = round(input[row * kInner + d]);
        a = std::fma(x, round(first[column * kInner + d]), a);
        b = std::fma(x, round(second[column * kInner + d]), b);
      }
      expected[row * kColumns + column] = round(a) + round(b) + bias[column];
    }
  }
  for (const auto &device : devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<float> x;
    test::BackendTensor<float> a;
    test::BackendTensor<float> b;
    test::BackendTensor<float> bias_tensor;
    test::BackendTensor<float> y;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(a.Initialize(device, first));
    ASSERT_TRUE(b.Initialize(device, second));
    ASSERT_TRUE(bias_tensor.Initialize(device, bias));
    ASSERT_TRUE(y.Initialize(device, std::vector<float>(expected.size())));
    ASSERT_TRUE(device->Compute(backend::ScalarType::kFloat32,
                                backend::PairedPatchEmbedding{.input = x.view(),
                                                              .first_weight = a.view(),
                                                              .second_weight = b.view(),
                                                              .bias = bias_tensor.view(),
                                                              .height = 6,
                                                              .width = 10,
                                                              .channels = 3,
                                                              .output_channels = kColumns,
                                                              .spatial_patch = 2,
                                                              .output = y.view(),
                                                              .is_fp16_projection = true}));
    std::vector<float> actual;
    ASSERT_TRUE(y.ReadFloats(&actual));
    test::ExpectBackendClose(actual, expected, {1.0e-7F, 0.0F});
  }
}

TYPED_TEST(VisionBackendTest, PatchPositionRopeAndBiasVariantsMatchBothBackends) {
  constexpr int32_t kGrid = 4;
  constexpr int32_t kTokens = kGrid * kGrid;
  constexpr int32_t kHidden = 80;
  constexpr int32_t kInner = 12;
  const auto patches = test::BackendInput<TypeParam>(kTokens * kInner, 1);
  const auto weight = test::BackendInput<TypeParam>(kHidden * kInner, 3);
  const auto second = test::BackendInput<TypeParam>(weight.size(), 5);
  const auto bias = test::BackendInput<TypeParam>(kHidden, 7);
  const auto position = test::BackendInput<TypeParam>(9U * kHidden, 2);
  const auto qkv = test::BackendInput<TypeParam>(kTokens * kHidden * 3U, 9);
  for (int32_t operation = 0; operation < 7; ++operation) {
    SCOPED_TRACE(operation);
    std::array<std::vector<float>, 2U> outputs;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
      test::BackendTensor<TypeParam> x;
      test::BackendTensor<TypeParam> w;
      test::BackendTensor<TypeParam> w2;
      test::BackendTensor<TypeParam> b;
      test::BackendTensor<TypeParam> p;
      test::BackendTensor<TypeParam> y;
      test::BackendTensor<TypeParam> k;
      test::BackendTensor<TypeParam> v;
      const auto input = operation == 0   ? patches
                         : operation == 2 ? qkv
                                          : test::BackendInput<TypeParam>(kTokens * kHidden, 4);
      ASSERT_TRUE(x.Initialize(this->devices()[device], input));
      ASSERT_TRUE(w.Initialize(this->devices()[device], weight));
      ASSERT_TRUE(w2.Initialize(this->devices()[device], second));
      ASSERT_TRUE(b.Initialize(this->devices()[device], bias));
      ASSERT_TRUE(p.Initialize(this->devices()[device], position));
      ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(kTokens * kHidden)));
      ASSERT_TRUE(k.Initialize(this->devices()[device], std::vector<TypeParam>(kTokens * kHidden)));
      ASSERT_TRUE(v.Initialize(this->devices()[device], std::vector<TypeParam>(kTokens * kHidden)));
      backend::ComputeOperation call;
      switch (operation) {
      case 0:
        call = backend::PairedPatchEmbedding{.input = x.view(),
                                             .first_weight = w.view(),
                                             .second_weight = w2.view(),
                                             .bias = b.view(),
                                             .height = 8,
                                             .width = 8,
                                             .channels = 3,
                                             .output_channels = kHidden,
                                             .spatial_patch = 2,
                                             .output = y.view()};
        break;
      case 1:
        call = backend::GridPositionEmbedding{.input = x.view(),
                                              .position_weight = p.view(),
                                              .grid_height = kGrid,
                                              .grid_width = kGrid,
                                              .position_grid_size = 3,
                                              .hidden_size = kHidden,
                                              .merge_size = 2,
                                              .output = y.view()};
        break;
      case 2:
        call = backend::GridQkvRope{.qkv = x.view(),
                                    .token_count = kTokens,
                                    .grid_height = kGrid,
                                    .grid_width = kGrid,
                                    .merge_size = 2,
                                    .head_count = 2,
                                    .head_size = 40,
                                    .query = y.view(),
                                    .key = k.view(),
                                    .value = v.view()};
        break;
      case 3:
        call = backend::Bias{.input = x.view(),
                             .bias = b.view(),
                             .element_count = input.size(),
                             .row_size = kHidden,
                             .output = x.view()};
        break;
      case 4:
        call = backend::BiasResidual{.input = x.view(),
                                     .bias = b.view(),
                                     .residual = x.view(),
                                     .element_count = input.size(),
                                     .row_size = kHidden,
                                     .output = y.view()};
        break;
      default:
        call = backend::GeluBias{.input = x.view(),
                                 .bias = b.view(),
                                 .element_count = input.size(),
                                 .row_size = kHidden,
                                 .is_approximate = operation == 5,
                                 .output = y.view()};
        break;
      }
      ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(), call));
      ASSERT_TRUE(operation == 3 ? x.ReadFloats(&outputs[device]) : y.ReadFloats(&outputs[device]));
      if (operation == 2) {
        std::vector<float> extra;
        if constexpr (std::is_same_v<TypeParam, float>) {
          test::BackendTensor<Float16> packed_key;
          test::BackendTensor<Float16> packed_value;
          ASSERT_TRUE(packed_key.Initialize(this->devices()[device],
                                            std::vector<Float16>(kTokens * kHidden)));
          ASSERT_TRUE(packed_value.Initialize(this->devices()[device],
                                              std::vector<Float16>(kTokens * kHidden)));
          auto packed_args = std::get<backend::GridQkvRope>(call);
          packed_args.key = packed_key.view();
          packed_args.value = packed_value.view();
          packed_args.is_fp16_key_value = true;
          ASSERT_TRUE(this->devices()[device]->Compute(backend::ScalarType::kFloat32, packed_args));
          std::vector<float> packed;
          std::vector<float> original;
          ASSERT_TRUE(packed_key.ReadFloats(&packed));
          ASSERT_TRUE(k.ReadFloats(&original));
          for (size_t element = 0U; element < packed.size(); ++element) {
            EXPECT_EQ(packed[element], test::BackendRound<Float16>(original[element]));
          }
          ASSERT_TRUE(packed_value.ReadFloats(&packed));
          ASSERT_TRUE(v.ReadFloats(&original));
          for (size_t element = 0U; element < packed.size(); ++element) {
            EXPECT_EQ(packed[element], test::BackendRound<Float16>(original[element]));
          }
          packed_args.value.size_bytes -= sizeof(uint16_t);
          EXPECT_FALSE(
              this->devices()[device]->Compute(backend::ScalarType::kFloat32, packed_args));
        }
        ASSERT_TRUE(k.ReadFloats(&extra));
        outputs[device].insert(outputs[device].end(), extra.begin(), extra.end());
        ASSERT_TRUE(v.ReadFloats(&extra));
        outputs[device].insert(outputs[device].end(), extra.begin(), extra.end());
      }
    }
    test::ExpectBackendClose(outputs[0], outputs[1], test::BackendMathTolerance<TypeParam>());
  }
}

TYPED_TEST(VisionBackendTest, TemporalPatchEmbeddingAndMergePreserveLayout) {
  constexpr int32_t kInner = 24;
  constexpr int32_t kPatches = 8;
  constexpr int32_t kColumns = 7;
  const auto input = test::BackendInput<TypeParam>(kPatches * kInner, 2);
  const auto weight = test::BackendInput<TypeParam>(kColumns * kInner, 3);
  const auto bias = test::BackendInput<TypeParam>(kColumns, 4);
  std::array<std::vector<float>, 2U> outputs;
  for (size_t device = 0U; device < this->devices().size(); ++device) {
    test::BackendTensor<TypeParam> x;
    test::BackendTensor<TypeParam> w;
    test::BackendTensor<TypeParam> b;
    test::BackendTensor<TypeParam> y;
    test::BackendTensor<TypeParam> merged;
    ASSERT_TRUE(x.Initialize(this->devices()[device], input));
    ASSERT_TRUE(w.Initialize(this->devices()[device], weight));
    ASSERT_TRUE(b.Initialize(this->devices()[device], bias));
    ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(kPatches * kColumns)));
    ASSERT_TRUE(
        merged.Initialize(this->devices()[device], std::vector<TypeParam>(kPatches * kColumns)));
    ASSERT_TRUE(this->devices()[device]->Compute(
        test::BackendType<TypeParam>(), backend::PatchEmbedding{.input = x.view(),
                                                                .weight = w.view(),
                                                                .bias = b.view(),
                                                                .frames = 4,
                                                                .height = 4,
                                                                .width = 4,
                                                                .channels = 3,
                                                                .output_channels = kColumns,
                                                                .temporal_patch = 2,
                                                                .spatial_patch = 2,
                                                                .output = y.view()}));
    ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                 backend::PatchMerge{.input = y.view(),
                                                                     .token_count = kPatches,
                                                                     .grid_width = 4,
                                                                     .hidden_size = kColumns,
                                                                     .merge_size = 2,
                                                                     .output = merged.view()}));
    ASSERT_TRUE(merged.ReadFloats(&outputs[device]));
    std::vector<float> original;
    ASSERT_TRUE(y.ReadFloats(&original));
    test::ExpectBackendClose(outputs[device], original, {0.0F, 0.0F});
  }
  test::ExpectBackendClose(outputs[0], outputs[1], test::BackendMathTolerance<TypeParam>());
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
