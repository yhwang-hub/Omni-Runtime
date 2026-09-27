#include <iostream>

#include "kernels/flash_attention_test_utils.h"
#include "kernels/flash_attention_v2_kernel/flash_attention_v2_cuda_runtime.h"
#include "kernels/flash_attention_v4_kernel/flash_attention_v4_cuda_runtime.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class FlashAttentionV4Test : public test::KernelTestFixture {};
using BlackwellTypes = ::testing::Types<Float16, BFloat16>;
TYPED_TEST_SUITE(FlashAttentionV4Test, BlackwellTypes);

TYPED_TEST(FlashAttentionV4Test, MatchesCpuForGroupedHeadsCausalMasksAndTailTiles) {
  if (!kernel::jit::JitRuntime::HasSm120()) {
    GTEST_SKIP() << "V4 SM120 attention requires Blackwell SM120/121";
  }
  test::CheckFlashAttention<TypeParam, runtime::FlashAttentionV4KernelRuntime>(this->backend());
}

TYPED_TEST(FlashAttentionV4Test, DISABLED_BenchmarksModelShapes) {
  if (!kernel::jit::JitRuntime::HasSm120()) {
    GTEST_SKIP() << "V4 SM120 attention requires Blackwell SM120/121";
  }
  for (const bool is_vision : {true, false}) {
    const int32_t tokens = is_vision ? 11008 : 2766;
    const int32_t heads = is_vision ? 16 : 32;
    const int32_t kv_heads = is_vision ? 16 : 8;
    const int32_t dimensions = is_vision ? 72 : 128;
    const size_t query_count = static_cast<size_t>(tokens) * heads * dimensions;
    const size_t kv_count = static_cast<size_t>(tokens) * kv_heads * dimensions;
    std::vector<TypeParam> query(query_count, test::LowPrecision<TypeParam>(0.125F));
    std::vector<TypeParam> key(kv_count, test::LowPrecision<TypeParam>(-0.25F));
    const auto q = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), query);
    const auto k = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), key);
    const auto v = memory::Buffer::CreateDeviceBufferFromVector(this->backend(), key);
    const auto output = memory::Buffer::CreateDeviceBuffer<TypeParam>(this->backend(), query_count);
    ASSERT_NE(q, nullptr);
    ASSERT_NE(k, nullptr);
    ASSERT_NE(v, nullptr);
    ASSERT_NE(output, nullptr);
    const runtime::AttentionArgs args = {.options = {},
                                         .query = q->device_address(),
                                         .key_cache = k->device_address(),
                                         .value_cache = v->device_address(),
                                         .query_token_count = tokens,
                                         .query_head_count = heads,
                                         .key_value_head_count = kv_heads,
                                         .cached_token_count = tokens,
                                         .start_position = 0,
                                         .head_dim = dimensions,
                                         .is_causal = !is_vision,
                                         .output = output->device_address()};
    backend::EventTimer timer;
    ASSERT_TRUE(timer.Initialize(this->backend()));
    for (const int32_t variant : {2, 41, 42, 43, 44, 45, 46}) {
      if ((is_vision && (variant == 43 || variant == 44)) ||
          (!is_vision && (variant == 45 || variant == 46))) {
        continue;
      }
      const auto launch = [&]() {
        if (variant == 2) {
          return runtime::FlashAttentionV2KernelRuntime<TypeParam>::CompileAndLaunch("bench_v2",
                                                                                     args);
        }
        if (variant == 41) {
          return runtime::FlashAttentionV4KernelRuntime<TypeParam>::template CompileAndLaunchTiled<
              1, 32, 64>("bench_v4_single", args);
        }
        if (variant == 42) {
          return runtime::FlashAttentionV4KernelRuntime<TypeParam>::template CompileAndLaunchTiled<
              2, 32, 64>("bench_v4_double", args);
        }
        if (variant == 43) {
          return runtime::FlashAttentionV4KernelRuntime<TypeParam>::template CompileAndLaunchTiled<
              2, 32, 32>("bench_v4_small_double", args);
        }
        if (variant == 44) {
          return runtime::FlashAttentionV4KernelRuntime<TypeParam>::template CompileAndLaunchTiled<
              1, 32, 32>("bench_v4_small_single", args);
        }
        if (variant == 45) {
          return runtime::FlashAttentionV4KernelRuntime<TypeParam>::template CompileAndLaunchTiled<
              2, 64, 64>("bench_v4_large_double", args);
        }
        return runtime::FlashAttentionV4KernelRuntime<TypeParam>::template CompileAndLaunchTiled<
            1, 64, 64>("bench_v4_large_single", args);
      };
      ASSERT_TRUE(launch());
      ASSERT_TRUE(this->Sync());
      ASSERT_TRUE(timer.Start());
      constexpr int32_t kIterations = 20;
      for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
        ASSERT_TRUE(launch());
      }
      double elapsed_ms = 0.0;
      ASSERT_TRUE(timer.Stop(&elapsed_ms));
      std::cout << (is_vision ? "vision" : "prefill") << " variant=" << variant
                << " average_ms=" << elapsed_ms / kIterations << std::endl;
    }
  }
}

} // namespace
} // namespace omni_runtime::kernels
