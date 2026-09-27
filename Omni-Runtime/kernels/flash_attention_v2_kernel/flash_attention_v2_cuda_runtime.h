#pragma once

#include "kernels/cuda_operator_runtime.h"

#include "kernels/flash_attention_v2_kernel/mma_attention_runtime.h"

namespace omni_runtime::kernels::runtime {

template <typename T> class FlashAttentionV2KernelRuntime final {
public:
  static constexpr int32_t kMaxHeadDim = 128;
  using Args = AttentionArgs;

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    return LaunchMmaAttention<T, 1, 32, 64>(
        tag, args, "kernels/flash_attention_v2_kernel/flash_attention_v2_kernel.cuh",
        "omni_runtime::kernels::flash_attention_v2_kernel::MmaAttentionImpl");
  }
};

} // namespace omni_runtime::kernels::runtime
