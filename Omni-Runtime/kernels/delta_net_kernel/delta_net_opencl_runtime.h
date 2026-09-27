#pragma once

#include "kernels/opencl_operator_runtime.h"
#include "kernels/delta_net_kernel/delta_net_runtime_utils.h"

namespace omni_runtime::kernels::runtime {
template <> struct OpenCLOperator<backend::CausalConv1d> final {
  static bool Make(const backend::CausalConv1d &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, ElementBytes(call->type)), false, ERROR,
                           "Invalid causal convolution buffers or dimensions");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("delta_net_kernel/delta_net_kernel.cl", "CausalConv1d",
                                        (static_cast<uint64_t>(args.channels) + 255U) / 256U, 256U,
                                        call),
                       false);
    BindOpenCLArguments(call, args.input, args.weight, args.state, args.token_count, args.channels,
                        args.kernel_size, args.output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::GatedDeltaNet> final {
  static bool Make(const backend::GatedDeltaNet &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, ElementBytes(call->type)), false, ERROR,
                           "Invalid gated delta net buffers or dimensions");
    call->source = "delta_net_kernel/delta_net_kernel.cl";
    call->entry = "GatedDeltaNet";
    call->dimensions = 2U;
    call->local_size = {128U, 1U, 1U};
    call->global_size = {static_cast<size_t>(args.value_head_count) * 128U,
                         static_cast<size_t>((args.head_size + 3) / 4), 1U};
    BindOpenCLArguments(call, args.qkv, args.alpha, args.beta, args.decay_weight, args.time_bias,
                        args.state, args.token_count, args.key_head_count, args.value_head_count,
                        args.head_size, args.epsilon, args.output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::SigmoidGate> final {
  static bool Make(const backend::SigmoidGate &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, ElementBytes(call->type)), false, ERROR,
                           "Invalid sigmoid gate buffers");
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("delta_net_kernel/delta_net_kernel.cl", "SigmoidGate",
                                        (args.element_count + 255U) / 256U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.input, args.gate, args.element_count, args.output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::PartialMultiAxisRope> final {
  static bool Make(const backend::PartialMultiAxisRope &args,
                   backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(call == nullptr, false);
    OMNI_RETURN_VAL_IF_LOG(!ValidateDelta(args, ElementBytes(call->type)), false, ERROR,
                           "Invalid partial multi-axis rotary buffers or dimensions");
    const uint64_t count = static_cast<uint64_t>(args.tokens) *
                           std::max(args.query_heads, args.key_value_heads) * args.head_dim;
    OMNI_RETURN_VAL_IF(!ConfigureOpenCL("delta_net_kernel/delta_net_kernel.cl",
                                        "PartialMultiAxisRope", (count + 255U) / 256U, 256U, call),
                       false);
    BindOpenCLArguments(call, args.query, args.key, args.position_ids, args.tokens,
                        args.query_heads, args.key_value_heads, args.head_dim, args.rotary_dim,
                        args.temporal_section, args.height_section, args.width_section,
                        std::pow(args.frequency_base, -2.0F / static_cast<float>(args.rotary_dim)),
                        args.query_output, args.key_output);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
