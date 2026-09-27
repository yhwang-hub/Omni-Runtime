#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/gated_attention_kernel/gated_attention_cuda_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kTokenCount = 8;
constexpr int32_t kHeadCount = 32;
constexpr int32_t kHeadSize = 128;
constexpr std::size_t kOutputCount = 32768U;
constexpr std::size_t kQueryGateCount = 65536U;

class GatedAttentionKernelTest : public test::KernelTestFixture {};

TEST_F(GatedAttentionKernelTest, SplitsQwen3VlGatedQuery) {
  std::vector<float> query_gate(kQueryGateCount);
  std::vector<float> expected_query(kOutputCount);
  std::vector<float> expected_gate(kOutputCount);
  for (std::size_t index = 0; index < kQueryGateCount; ++index) {
    query_gate[index] = 0.001f * static_cast<float>(index % kHeadSize) - 0.064f;
  }
  for (std::size_t index = 0; index < kOutputCount; ++index) {
    const std::size_t head_row = index / kHeadSize;
    const std::size_t dimension = index % kHeadSize;
    const std::size_t source = head_row * 2U * kHeadSize + dimension;
    expected_query[index] = query_gate[source];
    expected_gate[index] = query_gate[source + kHeadSize];
  }
  const auto query_gate_device =
      memory::Buffer::CreateDeviceBufferFromVector(backend(), query_gate);
  const auto query_device =
      memory::Buffer::CreateDeviceBuffer<float>(backend(), expected_query.size());
  const auto gate_device =
      memory::Buffer::CreateDeviceBuffer<float>(backend(), expected_gate.size());
  ASSERT_TRUE((runtime::SplitGatedQueryKernelRuntime<float>::CompileAndLaunch(
      "split_gated_query_float", {.options = {.grid_dim = dim3(32, 1, 1),
                                              .block_dim = dim3(256, 1, 1),
                                              .shared_memory_bytes = 0},
                                  .query_gate = query_gate_device->device_address(),
                                  .token_count = kTokenCount,
                                  .head_count = kHeadCount,
                                  .head_size = kHeadSize,
                                  .query = query_device->device_address(),
                                  .gate = gate_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual_query(expected_query.size());
  std::vector<float> actual_gate(expected_gate.size());
  ASSERT_TRUE(query_device->CopyDataToVector(&actual_query));
  ASSERT_TRUE(gate_device->CopyDataToVector(&actual_gate));
  EXPECT_TRUE(test::IsClose(actual_query, expected_query, 1.0e-6f) &&
              test::IsClose(actual_gate, expected_gate, 1.0e-6f));
}
} // namespace
} // namespace omni_runtime::kernels

// Backend parity and validation checks exercise the operator runtime dispatch.
#include "kernels/backend_test_utils.h"
#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/gated_attention_kernel/gated_attention_opencl_runtime.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class SplitGatedQueryBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(SplitGatedQueryBackendTest, test::BackendTypes);

TYPED_TEST(SplitGatedQueryBackendTest, SplitsEachHeadInsteadOfWholeTokenAtNonzeroOffsets) {
  constexpr int32_t kTokens = 3;
  constexpr int32_t kHeads = 5;
  constexpr int32_t kSize = 37;
  constexpr size_t kCount = kTokens * kHeads * kSize;
  const auto input = test::BackendInput<TypeParam>(2U * kCount, 6);
  std::vector<float> expected_query(kCount), expected_gate(kCount);
  for (size_t index = 0U; index < kCount; ++index) {
    const size_t source = (index / kSize) * (2U * kSize) + index % kSize;
    expected_query[index] = test::BackendFloat(input[source]);
    expected_gate[index] = test::BackendFloat(input[source + kSize]);
  }
  std::array<std::vector<float>, 2U> queries, gates;
  for (size_t device = 0U; device < this->devices().size(); ++device) {
    SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
    test::BackendTensor<TypeParam> x, q, g;
    ASSERT_TRUE(x.Initialize(this->devices()[device], input));
    ASSERT_TRUE(q.Initialize(this->devices()[device], std::vector<TypeParam>(kCount)));
    ASSERT_TRUE(g.Initialize(this->devices()[device], std::vector<TypeParam>(kCount)));
    ASSERT_TRUE(this->devices()[device]->Compute(test::BackendType<TypeParam>(),
                                                 backend::SplitGatedQuery{.query_gate = x.view(),
                                                                          .token_count = kTokens,
                                                                          .head_count = kHeads,
                                                                          .head_size = kSize,
                                                                          .query = q.view(),
                                                                          .gate = g.view()}));
    ASSERT_TRUE(q.ReadFloats(&queries[device]));
    ASSERT_TRUE(g.ReadFloats(&gates[device]));
    test::ExpectBackendClose(queries[device], expected_query, {0.0F, 0.0F});
    test::ExpectBackendClose(gates[device], expected_gate, {0.0F, 0.0F});
  }
  test::ExpectBackendClose(queries[0], queries[1], {0.0F, 0.0F});
  test::ExpectBackendClose(gates[0], gates[1], {0.0F, 0.0F});
}

TEST(SplitGatedQueryValidationTest, RejectsOverflowAndMissingBuffers) {
  backend::OpenCLInvocation call;
  backend::SplitGatedQuery operation{};
  EXPECT_FALSE(runtime::OpenCLOperator<backend::SplitGatedQuery>::Make(operation, nullptr));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::SplitGatedQuery>::Make(operation, &call));
  operation.token_count = operation.head_count = operation.head_size =
      std::numeric_limits<int32_t>::max();
  EXPECT_FALSE(runtime::OpenCLOperator<backend::SplitGatedQuery>::Make(operation, &call));
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
