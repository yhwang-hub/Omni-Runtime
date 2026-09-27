#pragma once

#include "kernels/operator_runtime.h"

#include "kernels/flash_attention_v2_kernel/mma_attention_runtime.h"

namespace omni_runtime::kernels::runtime {

template <typename T> class FlashAttentionV4KernelRuntime final {
public:
  static constexpr int32_t kMaxHeadDim = 128;
  using Args = AttentionArgs;

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    return args.head_dim <= 80 ? CompileAndLaunchTiled<2, 64, 32>(tag, args)
                               : CompileAndLaunchTiled<1, 64, 32>(tag, args);
  }

  template <int32_t kStages, int32_t kSmallKeys, int32_t kLargeKeys>
  static bool CompileAndLaunchTiled(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF_LOG(!kernel::jit::JitRuntime::HasSm120(), false, ERROR,
                           "V4 SM120 attention requires a Blackwell SM120/SM121 device");
    return LaunchMmaAttention<T, kStages, kSmallKeys, kLargeKeys>(
        tag, args, "kernels/flash_attention_v4_kernel/flash_attention_v4_kernel.cuh",
        "omni_runtime::kernels::flash_attention_v4_kernel::BlackwellAttentionImpl");
  }
};

} // namespace omni_runtime::kernels::runtime
