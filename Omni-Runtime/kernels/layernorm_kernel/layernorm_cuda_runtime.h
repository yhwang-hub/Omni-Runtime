#pragma once

#include "kernels/cuda_operator_runtime.h"

#include <cstdint>
#include <string>

#include "kernels/kernel_runtime.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

template <typename T> class RmsNormKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const weight;
    int32_t rows;
    int32_t hidden_size;
    float epsilon;
    void *const output;
    bool is_ggml_normalization = false;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.weight == nullptr || args.output == nullptr,
                       false);
    const std::string include = "kernels/layernorm_kernel/layernorm_kernel.cuh";
    const std::string kernel = "omni_runtime::kernels::layernorm_kernel::RmsNormImpl";
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        args.is_ggml_normalization ? CudaKernelRuntime::MakeKernelSource<T, 1>(include, kernel)
                                   : CudaKernelRuntime::MakeKernelSource<T, 0>(include, kernel),
        args.options, static_cast<const T *>(args.input), static_cast<const T *>(args.weight),
        args.rows, args.hidden_size, args.epsilon, static_cast<T *>(args.output));
  }
};

template <typename T> class LayernormKernelRuntime final {
public:
  struct Args final {
    KernelLaunchConfig options;
    const void *const input;
    const void *const weight;
    const void *const bias;
    int32_t rows;
    int32_t hidden_size;
    float epsilon;
    void *const output;
    bool is_ggml_normalization = false;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.weight == nullptr || args.bias == nullptr ||
                           args.output == nullptr,
                       false);
    const std::string include = "kernels/layernorm_kernel/layernorm_kernel.cuh";
    const std::string kernel = "omni_runtime::kernels::layernorm_kernel::LayernormImpl";
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        args.is_ggml_normalization ? CudaKernelRuntime::MakeKernelSource<T, 1>(include, kernel)
                                   : CudaKernelRuntime::MakeKernelSource<T, 0>(include, kernel),
        args.options, static_cast<const T *>(args.input), static_cast<const T *>(args.weight),
        static_cast<const T *>(args.bias), args.rows, args.hidden_size, args.epsilon,
        static_cast<T *>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::RmsNorm> final {
  static bool Launch(const backend::RmsNorm &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.input.IsValid() || !args.weight.IsValid() ||
                               !args.output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.hidden_size <= 0 ||
                               (args.is_ggml_normalization && !std::is_same_v<T, float>),
                           false, ERROR, "Invalid compute operation dimensions");
    const uint32_t threads = args.is_ggml_normalization && args.hidden_size >= 1024 ? 1024U : 256U;
    return RmsNormKernelRuntime<T>::CompileAndLaunch(
        "backend_RmsNorm", {.options = {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
                                        .block_dim = {threads, 1U, 1U},
                                        .shared_memory_bytes = 0},
                            .input = CudaAddress<void>(args.input),
                            .weight = CudaAddress<void>(args.weight),
                            .rows = args.rows,
                            .hidden_size = args.hidden_size,
                            .epsilon = args.epsilon,
                            .output = CudaAddress<void>(args.output),
                            .is_ggml_normalization = args.is_ggml_normalization});
  }
};

template <typename T> struct CudaOperator<T, backend::ResidualRmsNorm> final {
  static bool Launch(const backend::ResidualRmsNorm &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.input.IsValid() || !args.residual.IsValid() ||
                               !args.weight.IsValid() || !args.residual_output.IsValid() ||
                               !args.output.IsValid() ||
                               args.input.backend_kind != backend::BackendKind::kCuda ||
                               args.residual.backend_kind != backend::BackendKind::kCuda ||
                               args.weight.backend_kind != backend::BackendKind::kCuda ||
                               args.residual_output.backend_kind != backend::BackendKind::kCuda ||
                               args.output.backend_kind != backend::BackendKind::kCuda ||
                               args.rows <= 0 || args.hidden_size <= 0,
                           false, ERROR, "Invalid residual RMSNorm inputs");
    OMNI_RETURN_VAL_IF_LOG((args.is_ggml_normalization && !std::is_same_v<T, float>), false, ERROR,
                           "GGML normalization requires FP32 storage");
    const uint32_t threads = args.is_ggml_normalization && args.hidden_size >= 1024 ? 1024U : 256U;
    const std::string include = "kernels/layernorm_kernel/layernorm_kernel.cuh";
    const std::string kernel = "omni_runtime::kernels::layernorm_kernel::ResidualRmsNormImpl";
    return CudaKernelRuntime::CompileAndLaunch(
        "backend_ResidualRmsNorm",
        args.is_ggml_normalization ? CudaKernelRuntime::MakeKernelSource<T, 1>(include, kernel)
                                   : CudaKernelRuntime::MakeKernelSource<T, 0>(include, kernel),
        {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
         .block_dim = {threads, 1U, 1U},
         .shared_memory_bytes = 0},
        CudaAddress<const T>(args.input), CudaAddress<const T>(args.residual),
        CudaAddress<const T>(args.weight), args.rows, args.hidden_size, args.epsilon,
        CudaAddress<T>(args.residual_output), CudaAddress<T>(args.output));
  }
};

template <typename T> struct CudaOperator<T, backend::LayerNorm> final {
  static bool Launch(const backend::LayerNorm &args) {
    OMNI_RETURN_VAL_IF_LOG(!args.input.IsValid() || !args.weight.IsValid() ||
                               !args.bias.IsValid() || !args.output.IsValid(),
                           false, ERROR, "Invalid CUDA operator views");
    OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.hidden_size <= 0 ||
                               (args.is_ggml_normalization && !std::is_same_v<T, float>),
                           false, ERROR, "Invalid compute operation dimensions");
    const uint32_t threads =
        args.is_ggml_normalization ? (args.hidden_size < 1024 ? 32U : 1024U) : 256U;
    return LayernormKernelRuntime<T>::CompileAndLaunch(
        "backend_LayerNorm", {.options = {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
                                          .block_dim = {threads, 1U, 1U},
                                          .shared_memory_bytes = 0},
                              .input = CudaAddress<void>(args.input),
                              .weight = CudaAddress<void>(args.weight),
                              .bias = CudaAddress<void>(args.bias),
                              .rows = args.rows,
                              .hidden_size = args.hidden_size,
                              .epsilon = args.epsilon,
                              .output = CudaAddress<void>(args.output),
                              .is_ggml_normalization = args.is_ggml_normalization});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
