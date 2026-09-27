#include "kernels/backend_test_utils.h"

#include <algorithm>
#include <limits>

namespace omni_runtime::kernels {
namespace {

template <typename T> class AttentionBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(AttentionBackendTest, test::BackendTypes);

TYPED_TEST(AttentionBackendTest, GroupedCausalPrefillAndResidentDecodeMatchReference) {
  for (const auto shape : {std::array<int32_t, 2U>{4, 72}, std::array<int32_t, 2U>{8, 128}}) {
    const int32_t kQueryHeads = shape[0];
    constexpr int32_t kKvHeads = 2;
    const int32_t kHead = shape[1];
    for (const int32_t tokens : {1, 7}) {
      const int32_t cached = tokens == 1 ? 257 : 9;
      const int32_t start = tokens == 1 ? cached - 1 : 2;
      const auto query = test::BackendInput<TypeParam>(tokens * kQueryHeads * kHead, 2);
      const auto key = test::BackendInput<TypeParam>(cached * kKvHeads * kHead, 4);
      const auto value = test::BackendInput<TypeParam>(key.size(), 9);
      std::vector<float> expected(query.size(), 0.0F);
      for (int32_t token = 0; token < tokens; ++token) {
        const int32_t limit = std::min(cached, start + token + 1);
        for (int32_t head = 0; head < kQueryHeads; ++head) {
          std::vector<double> scores(limit);
          double maximum = -std::numeric_limits<double>::infinity();
          const int32_t q = (token * kQueryHeads + head) * kHead;
          for (int32_t source = 0; source < limit; ++source) {
            const int32_t k = (source * kKvHeads + head / (kQueryHeads / kKvHeads)) * kHead;
            double sum = 0.0;
            for (int32_t d = 0; d < kHead; ++d) {
              sum += test::BackendFloat(query[q + d]) * test::BackendFloat(key[k + d]);
            }
            scores[source] = sum / std::sqrt(kHead);
            maximum = std::max(maximum, scores[source]);
          }
          double denominator = 0.0;
          for (double &score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
          }
          for (int32_t d = 0; d < kHead; ++d) {
            double sum = 0.0;
            for (int32_t source = 0; source < limit; ++source) {
              sum += scores[source] / denominator *
                     test::BackendFloat(
                         value[(source * kKvHeads + head / (kQueryHeads / kKvHeads)) * kHead + d]);
            }
            expected[q + d] = test::BackendRound<TypeParam>(sum);
          }
        }
      }
      std::array<std::vector<float>, 2U> outputs;
      for (size_t device = 0U; device < this->devices().size(); ++device) {
        SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
        test::BackendTensor<TypeParam> q;
        test::BackendTensor<TypeParam> k;
        test::BackendTensor<TypeParam> v;
        test::BackendTensor<TypeParam> y;
        test::BackendTensor<int32_t> position;
        test::BackendTensor<float> partials;
        ASSERT_TRUE(q.Initialize(this->devices()[device], query));
        ASSERT_TRUE(k.Initialize(this->devices()[device], key));
        ASSERT_TRUE(v.Initialize(this->devices()[device], value));
        ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(query.size())));
        ASSERT_TRUE(position.Initialize(this->devices()[device], {start}));
        ASSERT_TRUE(partials.Initialize(this->devices()[device],
                                        std::vector<float>(3U * kQueryHeads * (kHead + 2U))));
        ASSERT_TRUE(this->devices()[device]->Compute(
            test::BackendType<TypeParam>(),
            backend::Attention{
                .query = q.view(),
                .key_cache = k.view(),
                .value_cache = v.view(),
                .query_token_count = tokens,
                .query_head_count = kQueryHeads,
                .key_value_head_count = kKvHeads,
                .cached_token_count = cached,
                .start_position = start,
                .head_dim = kHead,
                .is_causal = true,
                .output = y.view(),
                .decode_partials = tokens == 1 ? partials.view() : backend::DeviceView{},
                .cache_position = tokens == 1 ? position.view() : backend::DeviceView{}}));
        ASSERT_TRUE(y.ReadFloats(&outputs[device]));
        test::ExpectBackendClose(outputs[device], expected,
                                 test::BackendMathTolerance<TypeParam>());
      }
      test::ExpectBackendClose(outputs[0], outputs[1], test::BackendMathTolerance<TypeParam>());
    }
  }
}

TYPED_TEST(AttentionBackendTest, PreparedDecodeCrossesPartitionsAndResetsResidentState) {
  constexpr int32_t kCapacity = 385;
  constexpr int32_t kQueryHeads = 8;
  constexpr int32_t kKvHeads = 2;
  constexpr int32_t kHead = 128;
  constexpr size_t kOutputCount = kQueryHeads * kHead;
  const auto query = test::BackendInput<TypeParam>(kOutputCount, 3);
  const auto key = test::BackendInput<TypeParam>(kCapacity * kKvHeads * kHead, 5);
  const auto value = test::BackendInput<TypeParam>(key.size(), 8);
  for (const auto &device : this->devices()) {
    SCOPED_TRACE(backend::BackendName(device->kind()));
    test::BackendTensor<TypeParam> q;
    test::BackendTensor<TypeParam> k;
    test::BackendTensor<TypeParam> v;
    test::BackendTensor<TypeParam> y;
    test::BackendTensor<TypeParam> reference;
    test::BackendTensor<float> partials;
    ASSERT_TRUE(q.Initialize(device, query));
    ASSERT_TRUE(k.Initialize(device, key));
    ASSERT_TRUE(v.Initialize(device, value));
    ASSERT_TRUE(y.Initialize(device, std::vector<TypeParam>(kOutputCount)));
    ASSERT_TRUE(reference.Initialize(device, std::vector<TypeParam>(kOutputCount)));
    ASSERT_TRUE(
        partials.Initialize(device, std::vector<float>(4U * kQueryHeads * (kHead + 2U), 97.0F)));
    const auto state =
        memory::Buffer::CreateDeviceBufferFromVector(device, std::vector<int32_t>(4U, 0));
    ASSERT_NE(state, nullptr);
    backend::Attention operation{.query = q.view(),
                                 .key_cache = k.view(),
                                 .value_cache = v.view(),
                                 .query_token_count = 1,
                                 .query_head_count = kQueryHeads,
                                 .key_value_head_count = kKvHeads,
                                 .cached_token_count = kCapacity,
                                 .start_position = 0,
                                 .head_dim = kHead,
                                 .is_causal = true,
                                 .output = y.view(),
                                 .decode_partials = partials.view(),
                                 .cache_position =
                                     state->view().Slice(3U * sizeof(int32_t), sizeof(int32_t))};
    const backend::DeviceView output = y.view();
    const backend::Allocation readback{.data = output.data,
                                       .size_bytes = output.size_bytes,
                                       .memory_kind = backend::MemoryKind::kDeviceView,
                                       .offset_bytes = output.offset_bytes};
    const std::vector<backend::ComputeOperation> operations = {
        operation,
        backend::AdvanceSequence{.positions = state->view().Slice(0U, 3U * sizeof(int32_t)),
                                 .cache_position = operation.cache_position}};
    std::unique_ptr<backend::ExecutionPlan> plan;
    ASSERT_TRUE(
        device->CreateExecutionPlan(test::BackendType<TypeParam>(), operations, readback, &plan));
    ASSERT_NE(plan, nullptr);
    std::vector<int32_t> actual_state;
    ASSERT_TRUE(state->CopyDataToVector(&actual_state));
    EXPECT_EQ(actual_state, std::vector<int32_t>(4U, 0));
    for (const int32_t start : {126, 254, 383, 0}) {
      SCOPED_TRACE(start);
      const std::vector<int32_t> reset_state(4U, start);
      ASSERT_TRUE(state->CopyFromHost(reset_state.data(), reset_state.size() * sizeof(int32_t)));
      const int32_t steps = std::min(3, kCapacity - start);
      for (int32_t step = 0; step < steps; ++step) {
        const int32_t position = start + step;
        std::vector<TypeParam> result(kOutputCount);
        double elapsed_ms = -1.0;
        ASSERT_TRUE(plan->Run(result.size() * sizeof(TypeParam), result.data(), &elapsed_ms));
        EXPECT_GT(elapsed_ms, 0.0);
        backend::Attention direct = operation;
        direct.output = reference.view();
        direct.decode_partials = {};
        direct.cache_position = {};
        direct.cached_token_count = position + 1;
        direct.start_position = position;
        ASSERT_TRUE(device->Compute(test::BackendType<TypeParam>(), direct));
        std::vector<float> expected;
        ASSERT_TRUE(reference.ReadFloats(&expected));
        test::ExpectBackendClose(test::BackendFloats(result), expected,
                                 test::BackendMathTolerance<TypeParam>());
        ASSERT_TRUE(state->CopyDataToVector(&actual_state));
        EXPECT_EQ(actual_state, std::vector<int32_t>(4U, position + 1));
      }
    }
    std::vector<TypeParam> guarded_output;
    ASSERT_TRUE(y.Read(&guarded_output));
    std::vector<float> guarded_partials;
    ASSERT_TRUE(partials.Read(&guarded_partials));
  }
}

} // namespace
} // namespace omni_runtime::kernels
