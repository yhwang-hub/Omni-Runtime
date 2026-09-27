#pragma once

#include "kernels/cuda_operator_runtime.h"
#include "kernels/delta_net_kernel/delta_net_runtime_utils.h"

namespace omni_runtime::kernels::runtime {
template <typename T> struct CudaOperator<T, backend::CausalConv1d> final {
  static bool Launch(const backend::CausalConv1d &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, sizeof(T)), false, ERROR,
                           "Invalid causal convolution buffers or dimensions");
    OMNI_RETURN_VAL_IF(CudaAddress<const T>(args.input) == nullptr ||
                           CudaAddress<const float>(args.weight) == nullptr ||
                           CudaAddress<float>(args.state) == nullptr ||
                           CudaAddress<T>(args.output) == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        "causal_conv1d",
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/delta_net_kernel/delta_net_kernel.cuh",
            "omni_runtime::kernels::delta_net_kernel::CausalConvImpl"),
        {.grid_dim = {static_cast<uint32_t>((args.channels + 255) / 256), 1U, 1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = 0},
        CudaAddress<const T>(args.input), CudaAddress<const float>(args.weight),
        CudaAddress<float>(args.state), args.token_count, args.channels, args.kernel_size,
        CudaAddress<T>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::GatedDeltaNet> final {
  static bool Launch(const backend::GatedDeltaNet &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, sizeof(T)), false, ERROR,
                           "Invalid gated delta net buffers or dimensions");
    OMNI_RETURN_VAL_IF(
        CudaAddress<const T>(args.qkv) == nullptr || CudaAddress<const T>(args.alpha) == nullptr ||
            CudaAddress<const T>(args.beta) == nullptr ||
            CudaAddress<const float>(args.decay_weight) == nullptr ||
            CudaAddress<const float>(args.time_bias) == nullptr ||
            CudaAddress<float>(args.state) == nullptr || CudaAddress<T>(args.output) == nullptr,
        false);
    const bool is_prefill = args.token_count > 1;
    const int32_t columns_per_block = is_prefill ? 16 : 4;
    const std::string include = "kernels/delta_net_kernel/delta_net_kernel.cuh";
    const std::string kernel = "omni_runtime::kernels::delta_net_kernel::DeltaNetImpl";
    return CudaKernelRuntime::CompileAndLaunch(
        "gated_delta_net",
        is_prefill ? CudaKernelRuntime::MakeKernelSource<T, 4>(include, kernel)
                   : CudaKernelRuntime::MakeKernelSource<T, 1>(include, kernel),
        {.grid_dim = {static_cast<uint32_t>(args.value_head_count),
                      static_cast<uint32_t>((args.head_size + columns_per_block - 1) /
                                            columns_per_block),
                      1U},
         .block_dim = {128U, 1U, 1U},
         .shared_memory_bytes = 0},
        CudaAddress<const T>(args.qkv), CudaAddress<const T>(args.alpha),
        CudaAddress<const T>(args.beta), CudaAddress<const float>(args.decay_weight),
        CudaAddress<const float>(args.time_bias), CudaAddress<float>(args.state), args.token_count,
        args.key_head_count, args.value_head_count, args.head_size, args.epsilon,
        CudaAddress<T>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::SigmoidGate> final {
  static bool Launch(const backend::SigmoidGate &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, sizeof(T)), false, ERROR,
                           "Invalid sigmoid gate buffers");
    OMNI_RETURN_VAL_IF(CudaAddress<const T>(args.input) == nullptr ||
                           CudaAddress<const T>(args.gate) == nullptr ||
                           CudaAddress<T>(args.output) == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        "sigmoid_gate",
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/delta_net_kernel/delta_net_kernel.cuh",
            "omni_runtime::kernels::delta_net_kernel::SigmoidGateImpl"),
        {.grid_dim = {static_cast<uint32_t>(
                          std::min<uint64_t>((args.element_count + 255U) / 256U, 65535U)),
                      1U, 1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = 0},
        CudaAddress<const T>(args.input), CudaAddress<const T>(args.gate), args.element_count,
        CudaAddress<T>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::PartialMultiAxisRope> final {
  static bool Launch(const backend::PartialMultiAxisRope &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, sizeof(T)), false, ERROR,
                           "Invalid partial multi-axis rotary buffers or dimensions");
    OMNI_RETURN_VAL_IF(CudaAddress<const T>(args.query) == nullptr ||
                           CudaAddress<const T>(args.key) == nullptr ||
                           CudaAddress<const int32_t>(args.position_ids) == nullptr ||
                           CudaAddress<T>(args.query_output) == nullptr ||
                           CudaAddress<T>(args.key_output) == nullptr,
                       false);
    const uint64_t count = static_cast<uint64_t>(args.tokens) *
                           std::max(args.query_heads, args.key_value_heads) * args.head_dim;
    return CudaKernelRuntime::CompileAndLaunch(
        "partial_mrope",
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/delta_net_kernel/delta_net_kernel.cuh",
            "omni_runtime::kernels::delta_net_kernel::PartialRopeImpl"),
        {.grid_dim = {static_cast<uint32_t>(std::min<uint64_t>((count + 255U) / 256U, 65535U)), 1U,
                      1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = 0},
        CudaAddress<const T>(args.query), CudaAddress<const T>(args.key),
        CudaAddress<const int32_t>(args.position_ids), args.tokens, args.query_heads,
        args.key_value_heads, args.head_dim, args.rotary_dim, args.temporal_section,
        args.height_section, args.width_section,
        std::pow(args.frequency_base, -2.0F / static_cast<float>(args.rotary_dim)),
        CudaAddress<T>(args.query_output), CudaAddress<T>(args.key_output));
  }
};

} // namespace omni_runtime::kernels::runtime
