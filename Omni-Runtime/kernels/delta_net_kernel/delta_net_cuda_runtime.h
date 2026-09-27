#pragma once

#include "Omni-Runtime/backend/backend_cuda_utils.h"
#include "kernels/kernel_runtime.h"
#include "kernels/delta_net_kernel/delta_net_runtime_utils.h"

namespace omni_runtime::kernels::runtime {
template <typename T> struct CudaOperator<T, backend::CausalConv1d> final {
  static bool Launch(const backend::CausalConv1d &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, sizeof(T)), false, ERROR,
                           "Invalid causal convolution buffers or dimensions");
    OMNI_RETURN_VAL_IF(backend::cuda_internal::CudaAddress<const T>(args.input) == nullptr ||
                           backend::cuda_internal::CudaAddress<const float>(args.weight) ==
                               nullptr ||
                           backend::cuda_internal::CudaAddress<float>(args.state) == nullptr ||
                           backend::cuda_internal::CudaAddress<T>(args.output) == nullptr,
                       false);
    return CudaKernelRuntime::CompileAndLaunch(
        "causal_conv1d",
        CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/delta_net_kernel/delta_net_kernel.cuh",
            "omni_runtime::kernels::delta_net_kernel::CausalConvImpl"),
        {.grid_dim = {static_cast<uint32_t>((args.channels + 255) / 256), 1U, 1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = 0},
        backend::cuda_internal::CudaAddress<const T>(args.input),
        backend::cuda_internal::CudaAddress<const float>(args.weight),
        backend::cuda_internal::CudaAddress<float>(args.state), args.token_count, args.channels,
        args.kernel_size, backend::cuda_internal::CudaAddress<T>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::GatedDeltaNet> final {
  static bool Launch(const backend::GatedDeltaNet &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, sizeof(T)), false, ERROR,
                           "Invalid gated delta net buffers or dimensions");
    OMNI_RETURN_VAL_IF_LOG(args.is_log_decay && !args.is_chunked, false, ERROR,
                           "Logarithmic decay requires the reference delta net path");
    OMNI_RETURN_VAL_IF(
        backend::cuda_internal::CudaAddress<const T>(args.qkv) == nullptr ||
            backend::cuda_internal::CudaAddress<const T>(args.alpha) == nullptr ||
            backend::cuda_internal::CudaAddress<const T>(args.beta) == nullptr ||
            backend::cuda_internal::CudaAddress<const float>(args.decay_weight) == nullptr ||
            backend::cuda_internal::CudaAddress<const float>(args.time_bias) == nullptr ||
            backend::cuda_internal::CudaAddress<float>(args.state) == nullptr ||
            backend::cuda_internal::CudaAddress<T>(args.output) == nullptr,
        false);
    if (args.is_chunked) {
      OMNI_RETURN_VAL_IF_LOG(args.token_count != 1 || args.head_size != 128, false, ERROR,
                             "Reference delta decode requires one token and 128-value heads");
      return CudaKernelRuntime::CompileAndLaunch(
          "reference_delta_decode",
          CudaKernelRuntime::MakeKernelSource<T>(
              "kernels/delta_net_kernel/delta_net_kernel.cuh",
              "omni_runtime::kernels::delta_net_kernel::ReferenceDeltaDecodeImpl"),
          {.grid_dim = {static_cast<uint32_t>(args.value_head_count), 1U, 1U},
           .block_dim = {32U, 4U, 1U},
           .shared_memory_bytes = 0},
          backend::cuda_internal::CudaAddress<const T>(args.qkv),
          backend::cuda_internal::CudaAddress<const T>(args.alpha),
          backend::cuda_internal::CudaAddress<const T>(args.beta),
          backend::cuda_internal::CudaAddress<const float>(args.decay_weight),
          backend::cuda_internal::CudaAddress<const float>(args.time_bias), args.key_head_count,
          args.value_head_count, args.epsilon, args.is_log_decay,
          backend::cuda_internal::CudaAddress<float>(args.state),
          backend::cuda_internal::CudaAddress<T>(args.output));
    }
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
        backend::cuda_internal::CudaAddress<const T>(args.qkv),
        backend::cuda_internal::CudaAddress<const T>(args.alpha),
        backend::cuda_internal::CudaAddress<const T>(args.beta),
        backend::cuda_internal::CudaAddress<const float>(args.decay_weight),
        backend::cuda_internal::CudaAddress<const float>(args.time_bias),
        backend::cuda_internal::CudaAddress<float>(args.state), args.token_count,
        args.key_head_count, args.value_head_count, args.head_size, args.epsilon,
        backend::cuda_internal::CudaAddress<T>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::SigmoidGate> final {
  static bool Launch(const backend::SigmoidGate &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, sizeof(T)), false, ERROR,
                           "Invalid sigmoid gate buffers");
    OMNI_RETURN_VAL_IF(backend::cuda_internal::CudaAddress<const T>(args.input) == nullptr ||
                           backend::cuda_internal::CudaAddress<const T>(args.gate) == nullptr ||
                           backend::cuda_internal::CudaAddress<T>(args.output) == nullptr,
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
        backend::cuda_internal::CudaAddress<const T>(args.input),
        backend::cuda_internal::CudaAddress<const T>(args.gate), args.element_count,
        backend::cuda_internal::CudaAddress<T>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::PartialMultiAxisRope> final {
  static bool Launch(const backend::PartialMultiAxisRope &args) {
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, sizeof(T)), false, ERROR,
                           "Invalid partial multi-axis rotary buffers or dimensions");
    OMNI_RETURN_VAL_IF(backend::cuda_internal::CudaAddress<const T>(args.query) == nullptr ||
                           backend::cuda_internal::CudaAddress<const T>(args.key) == nullptr ||
                           backend::cuda_internal::CudaAddress<const int32_t>(args.position_ids) ==
                               nullptr ||
                           backend::cuda_internal::CudaAddress<T>(args.query_output) == nullptr ||
                           backend::cuda_internal::CudaAddress<T>(args.key_output) == nullptr,
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
        backend::cuda_internal::CudaAddress<const T>(args.query),
        backend::cuda_internal::CudaAddress<const T>(args.key),
        backend::cuda_internal::CudaAddress<const int32_t>(args.position_ids), args.tokens,
        args.query_heads, args.key_value_heads, args.head_dim, args.rotary_dim,
        args.temporal_section, args.height_section, args.width_section,
        (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, backend::BFloat16>)
            ? args.frequency_base
            : std::pow(args.frequency_base, -2.0F / static_cast<float>(args.rotary_dim)),
        backend::cuda_internal::CudaAddress<T>(args.query_output),
        backend::cuda_internal::CudaAddress<T>(args.key_output));
  }
};

} // namespace omni_runtime::kernels::runtime
