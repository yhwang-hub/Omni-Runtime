#include <algorithm>

#include "kernels/backend_test_utils.h"
#include "kernels/delta_net_kernel/delta_net_cuda_runtime.h"
#include "kernels/delta_net_kernel/delta_net_opencl_runtime.h"
#include "kernels/delta_net_kernel/delta_net_runtime_utils.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class DeltaNetBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(DeltaNetBackendTest, test::BackendTypes);

TYPED_TEST(DeltaNetBackendTest, SigmoidGateMatchesReferenceInPlace) {
  const auto input = test::BackendInput<TypeParam>(513U, 2);
  const auto gate = test::BackendInput<TypeParam>(input.size(), 7);
  std::vector<float> expected(input.size());
  for (size_t index = 0U; index < input.size(); ++index) {
    expected[index] = test::BackendRound<TypeParam>(
        test::BackendFloat(input[index]) / (1.0F + std::exp(-test::BackendFloat(gate[index]))));
  }
  for (const auto &device : this->devices()) {
    test::BackendTensor<TypeParam> x;
    test::BackendTensor<TypeParam> g;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(g.Initialize(device, gate));
    ASSERT_TRUE(device->Compute(test::BackendType<TypeParam>(),
                                backend::SigmoidGate{x.view(), g.view(), input.size(), x.view()}));
    std::vector<float> actual;
    ASSERT_TRUE(x.ReadFloats(&actual));
    test::ExpectBackendClose(actual, expected, test::BackendMathTolerance<TypeParam>());
  }
}

TYPED_TEST(DeltaNetBackendTest, CausalConvolutionPreservesHistoryAcrossBatches) {
  constexpr int32_t kTokens = 7;
  constexpr int32_t kChannels = 35;
  constexpr int32_t kKernel = 4;
  const auto input = test::BackendInput<TypeParam>(kTokens * kChannels, 2);
  const auto weights = test::BackendInput<float>(kChannels * kKernel, 3);
  const auto initial = test::BackendInput<float>(kChannels * (kKernel - 1), 7);
  std::vector<float> state = initial;
  std::vector<float> expected(input.size());
  for (int32_t t = 0; t < kTokens; ++t) {
    for (int32_t channel = 0; channel < kChannels; ++channel) {
      float sum = 0.0F;
      for (int32_t j = 0; j < kKernel - 1; ++j) {
        sum = std::fma(state[channel * (kKernel - 1) + j], weights[channel * kKernel + j], sum);
      }
      const float current = test::BackendFloat(input[t * kChannels + channel]);
      sum = std::fma(current, weights[channel * kKernel + kKernel - 1], sum);
      expected[t * kChannels + channel] =
          test::BackendRound<TypeParam>(sum / (1.0F + std::exp(-sum)));
      for (int32_t j = 0; j < kKernel - 2; ++j) {
        state[channel * (kKernel - 1) + j] = state[channel * (kKernel - 1) + j + 1];
      }
      state[channel * (kKernel - 1) + kKernel - 2] = current;
    }
  }
  for (const auto &device : this->devices()) {
    test::BackendTensor<TypeParam> x;
    test::BackendTensor<float> w;
    test::BackendTensor<float> s;
    test::BackendTensor<TypeParam> y;
    ASSERT_TRUE(x.Initialize(device, input));
    ASSERT_TRUE(w.Initialize(device, weights));
    ASSERT_TRUE(s.Initialize(device, initial));
    ASSERT_TRUE(y.Initialize(device, std::vector<TypeParam>(input.size())));
    for (const int32_t first : {0, 3}) {
      const int32_t count = first == 0 ? 3 : 4;
      const size_t offset = first * kChannels * sizeof(TypeParam);
      const size_t bytes = count * kChannels * sizeof(TypeParam);
      ASSERT_TRUE(device->Compute(test::BackendType<TypeParam>(),
                                  backend::CausalConv1d{x.view().Slice(offset, bytes), w.view(),
                                                        s.view(), count, kChannels, kKernel,
                                                        y.view().Slice(offset, bytes)}));
    }
    std::vector<float> actual;
    ASSERT_TRUE(y.ReadFloats(&actual));
    test::ExpectBackendClose(actual, expected, test::BackendMathTolerance<TypeParam>());
    ASSERT_TRUE(s.ReadFloats(&actual));
    test::ExpectBackendClose(actual, state, {0.0F, 0.0F});
  }
}

TYPED_TEST(DeltaNetBackendTest, DeltaRuleMatchesCpuAndChunkedState) {
  constexpr int32_t kTokens = 5;
  constexpr int32_t kKeyHeads = 2;
  constexpr int32_t kValueHeads = 6;
  constexpr int32_t kSize = 128;
  constexpr int32_t kChannels = (kKeyHeads * 2 + kValueHeads) * kSize;
  constexpr float kEpsilon = 1.0e-6F;
  const auto qkv = test::BackendInput<TypeParam>(kTokens * kChannels, 1);
  const auto alpha = test::BackendInput<TypeParam>(kTokens * kValueHeads, 5);
  const auto beta = test::BackendInput<TypeParam>(kTokens * kValueHeads, 7);
  const std::vector<float> decay(kValueHeads, -0.7F);
  const std::vector<float> bias(kValueHeads, 0.1F);
  const auto initial = test::BackendInput<float>(kValueHeads * kSize * kSize, 9);
  std::vector<float> state = initial;
  std::vector<float> expected(kTokens * kValueHeads * kSize);
  for (int32_t t = 0; t < kTokens; ++t) {
    for (int32_t head = 0; head < kValueHeads; ++head) {
      const size_t start = t * kChannels + (head % kKeyHeads) * kSize;
      float queries[kSize] = {};
      float keys[kSize] = {};
      double query_square = 0.0;
      double key_square = 0.0;
      for (int32_t d = 0; d < kSize; ++d) {
        queries[d] = test::BackendFloat(qkv[start + d]);
        keys[d] = test::BackendFloat(qkv[start + kKeyHeads * kSize + d]);
        query_square += queries[d] * queries[d];
        key_square += keys[d] * keys[d];
      }
      for (int32_t d = 0; d < kSize; ++d) {
        queries[d] /= std::sqrt(static_cast<float>(query_square) + kEpsilon);
        keys[d] /= std::sqrt(static_cast<float>(key_square) + kEpsilon);
      }
      const int32_t gate = t * kValueHeads + head;
      const float a = test::BackendFloat(alpha[gate]) + bias[head];
      const float g = std::exp(decay[head] * std::log1p(std::exp(a)));
      const float b = 1.0F / (1.0F + std::exp(-test::BackendFloat(beta[gate])));
      for (int32_t column = 0; column < kSize; ++column) {
        const size_t matrix = (head * kSize + column) * kSize;
        double projection = 0.0;
        for (int32_t d = 0; d < kSize; ++d) {
          projection += state[matrix + d] * keys[d];
        }
        const float value =
            test::BackendFloat(qkv[t * kChannels + 2 * kKeyHeads * kSize + head * kSize + column]);
        const float delta = (value - g * static_cast<float>(projection)) * b;
        double result = 0.0;
        for (int32_t d = 0; d < kSize; ++d) {
          state[matrix + d] = std::fma(keys[d], delta, g * state[matrix + d]);
          result += state[matrix + d] * queries[d];
        }
        expected[t * kValueHeads * kSize + head * kSize + column] =
            test::BackendRound<TypeParam>(static_cast<float>(result) / std::sqrt(float(kSize)));
      }
    }
  }
  for (const auto &device : this->devices()) {
    for (const bool is_chunked : {false, true}) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      SCOPED_TRACE(is_chunked);
      test::BackendTensor<TypeParam> x;
      test::BackendTensor<TypeParam> a;
      test::BackendTensor<TypeParam> b;
      test::BackendTensor<float> g;
      test::BackendTensor<float> dt;
      test::BackendTensor<float> s;
      test::BackendTensor<TypeParam> y;
      ASSERT_TRUE(x.Initialize(device, qkv));
      ASSERT_TRUE(a.Initialize(device, alpha));
      ASSERT_TRUE(b.Initialize(device, beta));
      ASSERT_TRUE(g.Initialize(device, decay));
      ASSERT_TRUE(dt.Initialize(device, bias));
      ASSERT_TRUE(s.Initialize(device, initial));
      ASSERT_TRUE(y.Initialize(device, std::vector<TypeParam>(expected.size())));
      for (int32_t first = 0; first < kTokens;) {
        const int32_t count = is_chunked ? std::min(2, kTokens - first) : kTokens;
        const auto slice = [first, count](const backend::DeviceView &view, const int32_t stride) {
          return view.Slice(static_cast<size_t>(first) * stride * sizeof(TypeParam),
                            static_cast<size_t>(count) * stride * sizeof(TypeParam));
        };
        ASSERT_TRUE(device->Compute(
            test::BackendType<TypeParam>(),
            backend::GatedDeltaNet{slice(x.view(), kChannels), slice(a.view(), kValueHeads),
                                   slice(b.view(), kValueHeads), g.view(), dt.view(), s.view(),
                                   count, kKeyHeads, kValueHeads, kSize, kEpsilon,
                                   slice(y.view(), kValueHeads * kSize)}));
        first += count;
      }
      std::vector<float> actual;
      ASSERT_TRUE(y.ReadFloats(&actual));
      test::ExpectBackendClose(actual, expected, test::BackendMathTolerance<TypeParam>());
      ASSERT_TRUE(s.ReadFloats(&actual));
      test::ExpectBackendClose(actual, state, {2.0e-6F, 2.0e-5F});
    }
  }
}

TYPED_TEST(DeltaNetBackendTest, PrefillMatchesSingleTokenOutputsAndStateExactly) {
  constexpr int32_t kTokens = 17;
  constexpr int32_t kKeyHeads = 2;
  constexpr int32_t kValueHeads = 6;
  for (const int32_t head_size : {24, 64, 128}) {
    const int32_t channels = (2 * kKeyHeads + kValueHeads) * head_size;
    const auto input = test::BackendInput<TypeParam>(kTokens * channels, 1);
    const auto alpha = test::BackendInput<TypeParam>(kTokens * kValueHeads, 5);
    const auto beta = test::BackendInput<TypeParam>(kTokens * kValueHeads, 7);
    const auto initial = test::BackendInput<float>(kValueHeads * head_size * head_size, 9);
    for (const auto &device : this->devices()) {
      SCOPED_TRACE(backend::BackendName(device->kind()));
      SCOPED_TRACE(head_size);
      test::BackendTensor<TypeParam> x;
      test::BackendTensor<TypeParam> a;
      test::BackendTensor<TypeParam> b;
      test::BackendTensor<float> decay;
      test::BackendTensor<float> bias;
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(a.Initialize(device, alpha));
      ASSERT_TRUE(b.Initialize(device, beta));
      ASSERT_TRUE(decay.Initialize(device, std::vector<float>(kValueHeads, -0.7F)));
      ASSERT_TRUE(bias.Initialize(device, std::vector<float>(kValueHeads, 0.1F)));
      std::array<std::vector<float>, 2U> outputs;
      std::array<std::vector<float>, 2U> states;
      for (size_t mode = 0U; mode < outputs.size(); ++mode) {
        const int32_t count = mode == 0U ? kTokens : 1;
        test::BackendTensor<float> state;
        test::BackendTensor<TypeParam> output;
        ASSERT_TRUE(state.Initialize(device, initial));
        ASSERT_TRUE(
            output.Initialize(device, std::vector<TypeParam>(kTokens * kValueHeads * head_size)));
        for (int32_t first = 0; first < kTokens; first += count) {
          const auto slice = [first, count](const backend::DeviceView &view, const int32_t stride) {
            return view.Slice(static_cast<size_t>(first) * stride * sizeof(TypeParam),
                              static_cast<size_t>(count) * stride * sizeof(TypeParam));
          };
          ASSERT_TRUE(device->Compute(
              test::BackendType<TypeParam>(),
              backend::GatedDeltaNet{slice(x.view(), channels), slice(a.view(), kValueHeads),
                                     slice(b.view(), kValueHeads), decay.view(), bias.view(),
                                     state.view(), count, kKeyHeads, kValueHeads, head_size,
                                     1.0e-6F, slice(output.view(), kValueHeads * head_size)}));
        }
        ASSERT_TRUE(output.ReadFloats(&outputs[mode]));
        ASSERT_TRUE(state.ReadFloats(&states[mode]));
      }
      EXPECT_EQ(outputs[0], outputs[1]);
      EXPECT_EQ(states[0], states[1]);
    }
  }
}

TYPED_TEST(DeltaNetBackendTest, PartialRotaryLeavesNonRotaryDimensionsUnchanged) {
  constexpr int32_t kTokens = 3;
  constexpr int32_t kHeads = 2;
  constexpr int32_t kSize = 256;
  constexpr int32_t kRotary = 64;
  const auto input = test::BackendInput<TypeParam>(kTokens * kHeads * kSize, 3);
  const std::vector<int32_t> positions{0, 77, 78, 4, 30, 78, 4, 76, 78};
  std::vector<float> expected = test::BackendFloats(input);
  for (size_t index = 0U; index < input.size(); ++index) {
    const int32_t d = index % kSize;
    if (d >= kRotary) {
      continue;
    }
    const int32_t pair = d % (kRotary / 2);
    const int32_t axis = pair % 3 == 1 && pair < 33 ? 1 : pair % 3 == 2 && pair < 30 ? 2 : 0;
    const int32_t token = index / (kHeads * kSize);
    const float theta_scale = std::pow(10000000.0F, -2.0F / kRotary);
    const float angle = positions[axis * kTokens + token] * std::pow(theta_scale, float(pair));
    const size_t other = d < kRotary / 2 ? index + kRotary / 2 : index - kRotary / 2;
    const float sign = d < kRotary / 2 ? -1.0F : 1.0F;
    expected[index] =
        test::BackendRound<TypeParam>(test::BackendFloat(input[index]) * std::cos(angle) +
                                      sign * test::BackendFloat(input[other]) * std::sin(angle));
  }
  std::array<std::vector<float>, 2U> results;
  for (const bool is_in_place : {false, true}) {
    for (size_t device_index = 0U; device_index < this->devices().size(); ++device_index) {
      const auto &device = this->devices()[device_index];
      test::BackendTensor<TypeParam> x;
      test::BackendTensor<TypeParam> key;
      test::BackendTensor<int32_t> p;
      test::BackendTensor<TypeParam> y;
      test::BackendTensor<TypeParam> key_result;
      ASSERT_TRUE(x.Initialize(device, input));
      ASSERT_TRUE(key.Initialize(device, input));
      ASSERT_TRUE(p.Initialize(device, positions));
      ASSERT_TRUE(y.Initialize(device, std::vector<TypeParam>(input.size())));
      ASSERT_TRUE(key_result.Initialize(device, std::vector<TypeParam>(input.size())));
      auto &output = is_in_place ? x : y;
      auto &key_output = is_in_place ? key : key_result;
      ASSERT_TRUE(device->Compute(test::BackendType<TypeParam>(),
                                  backend::PartialMultiAxisRope{x.view(), key.view(), p.view(),
                                                                kTokens, kHeads, kHeads, kSize,
                                                                kRotary, 11, 11, 10, 10000000.0F,
                                                                output.view(), key_output.view()}));
      std::vector<float> actual;
      ASSERT_TRUE(output.ReadFloats(&actual));
      test::ExpectBackendClose(actual, expected, test::BackendMathTolerance<TypeParam>());
      std::vector<float> actual_key;
      ASSERT_TRUE(key_output.ReadFloats(&actual_key));
      EXPECT_EQ(actual, actual_key);
      for (size_t index = 0U; index < input.size(); ++index) {
        if (index % kSize >= kRotary) {
          EXPECT_EQ(actual[index], test::BackendFloat(input[index]));
        }
      }
      results[device_index] = std::move(actual);
    }
    EXPECT_EQ(results[0], results[1]);
  }
}

TEST(DeltaNetValidationTest, RejectsInvalidShapesAndMissingStorage) {
  backend::OpenCLInvocation call;
  EXPECT_FALSE(runtime::OpenCLOperator<backend::GatedDeltaNet>::Make({}, &call));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::CausalConv1d>::Make({}, &call));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::PartialMultiAxisRope>::Make({}, &call));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::SigmoidGate>::Make({}, &call));
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
