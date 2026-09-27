#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_test_utils.h"
#include "kernels/mha_kernel/mha_cuda_runtime.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kHeadCount = 32;
constexpr int32_t kTokenCount = 8;
constexpr int32_t kHeadDim = 128;
constexpr std::size_t kElementCount = 32768U;

class MhaKernelTest : public test::KernelTestFixture {
protected:
  static void MakeAttentionInput(std::vector<float> *const tensor) {
    tensor->assign(kElementCount, 0.0f);
    for (std::size_t index = 0; index < kElementCount; ++index) {
      (*tensor)[index] = 0.0005f * static_cast<float>(index % kHeadDim) - 0.032f;
    }
  }
};

TEST_F(MhaKernelTest, AttendsQwen3VlTextTokens) {
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  MakeAttentionInput(&query);
  MakeAttentionInput(&key);
  MakeAttentionInput(&value);
  std::vector<float> expected;
  ASSERT_TRUE(test::ReferenceAttention(query, key, value, kHeadCount, kTokenCount, kHeadDim, true,
                                       &expected));
  const auto query_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), query);
  const auto key_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), key);
  const auto value_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), value);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE((runtime::MhaKernelRuntime<float>::CompileAndLaunch(
      "mha_float", {.options = {.grid_dim = dim3(kHeadCount * kTokenCount, 1, 1),
                                .block_dim = dim3(128, 1, 1),
                                .shared_memory_bytes = 0},
                    .query = query_device->device_address(),
                    .key = key_device->device_address(),
                    .value = value_device->device_address(),
                    .heads = kHeadCount,
                    .tokens = kTokenCount,
                    .head_dim = kHeadDim,
                    .is_causal = true,
                    .output = output_device->device_address()})));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5f));
}

TEST_F(MhaKernelTest, HandlesHeadMajorTailTiles) {
  constexpr int32_t kHeads = 3;
  constexpr int32_t kTokens = 17;
  constexpr int32_t kDimensions = 77;
  std::vector<float> query(kHeads * kTokens * kDimensions);
  std::vector<float> key(query.size());
  std::vector<float> value(query.size());
  for (size_t index = 0U; index < query.size(); ++index) {
    query[index] = (static_cast<int32_t>(index % 31U) - 15) / 16.0F;
    key[index] = (static_cast<int32_t>(index % 29U) - 14) / 16.0F;
    value[index] = (static_cast<int32_t>(index % 37U) - 18) / 16.0F;
  }
  const auto q = memory::Buffer::CreateDeviceBufferFromVector(backend(), query);
  const auto k = memory::Buffer::CreateDeviceBufferFromVector(backend(), key);
  const auto v = memory::Buffer::CreateDeviceBufferFromVector(backend(), value);
  const auto output = memory::Buffer::CreateDeviceBuffer<float>(backend(), query.size());
  ASSERT_NE(q, nullptr);
  ASSERT_NE(k, nullptr);
  ASSERT_NE(v, nullptr);
  ASSERT_NE(output, nullptr);
  for (const bool is_causal : {false, true}) {
    std::vector<float> expected;
    ASSERT_TRUE(test::ReferenceAttention(query, key, value, kHeads, kTokens, kDimensions, is_causal,
                                         &expected));
    ASSERT_TRUE((runtime::MhaKernelRuntime<float>::CompileAndLaunch(
        "mha_tail", {.options = {},
                     .query = q->device_address(),
                     .key = k->device_address(),
                     .value = v->device_address(),
                     .heads = kHeads,
                     .tokens = kTokens,
                     .head_dim = kDimensions,
                     .is_causal = is_causal,
                     .output = output->device_address()})));
    std::vector<float> actual;
    ASSERT_TRUE(output->CopyDataToVector(&actual));
    EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-5F));
  }
}
} // namespace
} // namespace omni_runtime::kernels

// Backend parity and validation checks exercise the operator runtime dispatch.
#include "kernels/backend_test_utils.h"
#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/mha_kernel/mha_opencl_runtime.h"

namespace omni_runtime::kernels::backend_runtime_test {
namespace {

template <typename T> class MhaBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(MhaBackendTest, test::BackendTypes);

TYPED_TEST(MhaBackendTest, HeadMajorCausalAndBidirectionalTilesMatchCpu) {
  constexpr int32_t kHeads = 3;
  for (const int32_t tokens : {1, 19}) {
    for (const int32_t dimensions : {7, 37, 256}) {
      const size_t count = kHeads * tokens * dimensions;
      const auto query = test::BackendInput<TypeParam>(count, 1);
      const auto key = test::BackendInput<TypeParam>(count, 5);
      const auto value = test::BackendInput<TypeParam>(count, 13);
      for (const bool causal : {false, true}) {
        SCOPED_TRACE(tokens);
        SCOPED_TRACE(dimensions);
        SCOPED_TRACE(causal);
        std::vector<float> expected;
        ASSERT_TRUE(test::ReferenceAttention(test::BackendFloats(query), test::BackendFloats(key),
                                             test::BackendFloats(value), kHeads, tokens, dimensions,
                                             causal, &expected));
        for (auto &item : expected) {
          item = test::BackendRound<TypeParam>(item);
        }
        std::array<std::vector<float>, 2U> results;
        for (size_t device = 0U; device < this->devices().size(); ++device) {
          SCOPED_TRACE(backend::BackendName(this->devices()[device]->kind()));
          test::BackendTensor<TypeParam> q, k, v, y;
          ASSERT_TRUE(q.Initialize(this->devices()[device], query));
          ASSERT_TRUE(k.Initialize(this->devices()[device], key));
          ASSERT_TRUE(v.Initialize(this->devices()[device], value));
          ASSERT_TRUE(y.Initialize(this->devices()[device], std::vector<TypeParam>(count)));
          ASSERT_TRUE(this->devices()[device]->Compute(
              test::BackendType<TypeParam>(), backend::MultiHeadAttention{.query = q.view(),
                                                                          .key = k.view(),
                                                                          .value = v.view(),
                                                                          .heads = kHeads,
                                                                          .tokens = tokens,
                                                                          .head_dim = dimensions,
                                                                          .is_causal = causal,
                                                                          .output = y.view()}));
          ASSERT_TRUE(y.ReadFloats(&results[device]));
          test::ExpectBackendClose(results[device], expected,
                                   test::BackendMathTolerance<TypeParam>());
        }
        test::ExpectBackendClose(results[0], results[1], test::BackendMathTolerance<TypeParam>());
      }
    }
  }
}

TEST(MhaValidationTest, RejectsOversizedHeadsAndMissingBuffers) {
  backend::OpenCLInvocation call;
  backend::MultiHeadAttention operation{};
  EXPECT_FALSE(runtime::OpenCLOperator<backend::MultiHeadAttention>::Make(operation, nullptr));
  EXPECT_FALSE(runtime::OpenCLOperator<backend::MultiHeadAttention>::Make(operation, &call));
  operation.heads = operation.tokens = 1;
  operation.head_dim = 257;
  EXPECT_FALSE(runtime::OpenCLOperator<backend::MultiHeadAttention>::Make(operation, &call));
  operation.head_dim = 37;
  EXPECT_FALSE(runtime::OpenCLOperator<backend::MultiHeadAttention>::Make(operation, &call));
}

} // namespace
} // namespace omni_runtime::kernels::backend_runtime_test
