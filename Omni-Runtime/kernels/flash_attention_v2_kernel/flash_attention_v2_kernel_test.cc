#include "kernels/flash_attention_test_utils.h"
#include "kernels/flash_attention_v2_kernel/flash_attention_v2_kernel_runtime.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class FlashAttentionV2Test : public test::KernelTestFixture {};
using MmaTypes = ::testing::Types<Float16, BFloat16>;
TYPED_TEST_SUITE(FlashAttentionV2Test, MmaTypes);

TYPED_TEST(FlashAttentionV2Test, MatchesCpuForGroupedHeadsCausalMasksAndTailTiles) {
  if (!runtime::CudaKernelRuntime::HasTensorMma()) {
    GTEST_SKIP() << "MMA attention requires SM80+";
  }
  test::CheckFlashAttention<TypeParam, runtime::FlashAttentionV2KernelRuntime>(this->backend());
}

} // namespace
} // namespace omni_runtime::kernels
