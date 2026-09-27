#pragma once

#include "kernels/operator_runtime_common.h"

#include "kernels/opencl_operator_runtime.h"

namespace omni_runtime::kernels::runtime {

inline bool ConfigureNorm(const std::string &entry, const backend::DeviceView &input,
                          const backend::DeviceView &weight, const int32_t rows,
                          const int32_t hidden_size, const float epsilon,
                          const backend::DeviceView &output,
                          backend::OpenCLInvocation *const call) {
  OMNI_RETURN_VAL_IF(call == nullptr, false);
  const size_t bytes = ElementBytes(call->type);
  OMNI_RETURN_VAL_IF_LOG(rows <= 0 || hidden_size <= 0 || epsilon <= 0.0F ||
                             !HasElements(input, uint64_t(rows) * hidden_size, bytes) ||
                             !HasElements(weight, hidden_size, bytes) ||
                             !HasElements(output, uint64_t(rows) * hidden_size, bytes),
                         false, ERROR, "Invalid normalization operation");
  return ConfigureOpenCL("layernorm_kernel/layernorm_kernel.cl", entry, rows, 256U, call);
}

template <> struct OpenCLOperator<backend::RmsNorm> final {
  static bool Make(const backend::RmsNorm &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(!ConfigureNorm("RmsNorm", args.input, args.weight, args.rows,
                                      args.hidden_size, args.epsilon, args.output, call),
                       false);
    if (args.is_ggml_normalization) {
      OMNI_RETURN_VAL_IF_LOG(call->type != backend::ScalarType::kFloat32, false, ERROR,
                             "GGML normalization requires FP32 storage");
      call->build_defines.push_back("OMNI_GGML_NORMALIZATION");
    }
    BindOpenCLArguments(call, args.input, args.weight, args.rows, args.hidden_size, args.epsilon,
                        args.output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::LayerNorm> final {
  static bool Make(const backend::LayerNorm &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(!ConfigureNorm("LayerNorm", args.input, args.weight, args.rows,
                                      args.hidden_size, args.epsilon, args.output, call),
                       false);
    OMNI_RETURN_VAL_IF_LOG(!HasElements(args.bias, args.hidden_size, ElementBytes(call->type)),
                           false, ERROR, "Invalid layer normalization bias");
    if (args.is_ggml_normalization) {
      OMNI_RETURN_VAL_IF_LOG(call->type != backend::ScalarType::kFloat32, false, ERROR,
                             "GGML normalization requires FP32 storage");
      const size_t threads = args.hidden_size < 1024 ? 32U : 256U;
      call->local_size = {threads, 1U, 1U};
      call->global_size = {static_cast<size_t>(args.rows) * threads, 1U, 1U};
      call->build_defines.push_back("OMNI_GGML_NORMALIZATION");
      call->build_defines.push_back("OMNI_NORM_THREADS=" + std::to_string(threads));
    }
    BindOpenCLArguments(call, args.input, args.weight, args.bias, args.rows, args.hidden_size,
                        args.epsilon, args.output);
    return true;
  }
};

template <> struct OpenCLOperator<backend::ResidualRmsNorm> final {
  static bool Make(const backend::ResidualRmsNorm &args, backend::OpenCLInvocation *const call) {
    OMNI_RETURN_VAL_IF(!ConfigureNorm("ResidualRmsNorm", args.input, args.weight, args.rows,
                                      args.hidden_size, args.epsilon, args.output, call),
                       false);
    const uint64_t count = uint64_t(args.rows) * args.hidden_size;
    OMNI_RETURN_VAL_IF_LOG(!HasElements(args.residual, count, ElementBytes(call->type)) ||
                               !HasElements(args.residual_output, count, ElementBytes(call->type)),
                           false, ERROR, "Invalid residual normalization buffers");
    if (args.is_ggml_normalization) {
      OMNI_RETURN_VAL_IF_LOG(call->type != backend::ScalarType::kFloat32, false, ERROR,
                             "GGML normalization requires FP32 storage");
      call->build_defines.push_back("OMNI_GGML_NORMALIZATION");
    }
    BindOpenCLArguments(call, args.input, args.residual, args.weight, args.rows, args.hidden_size,
                        args.epsilon, args.residual_output, args.output);
    return true;
  }
};

} // namespace omni_runtime::kernels::runtime
