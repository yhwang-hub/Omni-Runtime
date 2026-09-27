#pragma once

#include "Omni-Runtime/backend/backend_cuda_utils.h"
#include "kernels/kernel_runtime.h"

#include <algorithm>
#include <cstdint>
#include <string>

namespace omni_runtime {
namespace kernels {
namespace runtime {

inline uint32_t BfloatRmsNormThreads(const int32_t rows, const int32_t hidden_size) {
  uint32_t width = 32U;
  uint32_t height = 1U;
  while (width < 512U && width * 2U <= static_cast<uint32_t>(hidden_size / 4)) {
    width *= 2U;
  }
  while (height < 16U && height * 2U <= static_cast<uint32_t>(rows)) {
    height *= 2U;
  }
  return std::min(width, 512U / height);
}

template <typename T> class RmsNormKernelRuntime final {
public:
  struct Args final {
    backend::KernelLaunchConfig options;
    const void *const input;
    const void *const weight;
    int32_t rows;
    int32_t hidden_size;
    float epsilon;
    void *const output;
    bool is_ggml_normalization = false;
    bool is_float_weight = false;
  };

  static bool CompileAndLaunch(const std::string &tag, const Args &args) {
    OMNI_RETURN_VAL_IF(args.input == nullptr || args.weight == nullptr || args.output == nullptr,
                       false);
    const std::string include = "kernels/layernorm_kernel/layernorm_kernel.cuh";
    if constexpr (std::is_same_v<T, backend::BFloat16>) {
      if (!args.is_ggml_normalization && args.hidden_size >= 128 && args.hidden_size <= 5120 &&
          args.hidden_size % 4 == 0) {
        auto options = args.options;
        options.block_dim = {BfloatRmsNormThreads(args.rows, args.hidden_size), 1U, 1U};
        return CudaKernelRuntime::CompileAndLaunch(
            tag,
            CudaKernelRuntime::MakeKernelSource<T>(
                include, "omni_runtime::kernels::layernorm_kernel::BfloatRmsNormImpl"),
            options, static_cast<const T *>(args.input), static_cast<const T *>(nullptr),
            args.weight, args.rows, args.hidden_size, args.epsilon, args.is_float_weight,
            static_cast<T *>(nullptr), static_cast<T *>(args.output));
      }
    }
    const std::string kernel = "omni_runtime::kernels::layernorm_kernel::RmsNormImpl";
    return CudaKernelRuntime::CompileAndLaunch(
        tag,
        args.is_ggml_normalization ? CudaKernelRuntime::MakeKernelSource<T, 1>(include, kernel)
                                   : CudaKernelRuntime::MakeKernelSource<T, 0>(include, kernel),
        args.options, static_cast<const T *>(args.input), args.weight, args.rows, args.hidden_size,
        args.epsilon, static_cast<T *>(args.output), args.is_float_weight);
  }
};

template <typename T> class LayernormKernelRuntime final {
public:
  struct Args final {
    backend::KernelLaunchConfig options;
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
    if constexpr (std::is_same_v<T, backend::BFloat16>) {
      if (!args.is_ggml_normalization && args.hidden_size > 0 && args.hidden_size % 4 == 0) {
        uint32_t threads = 128U;
        while (threads > 32U && threads / 2U >= static_cast<uint32_t>(args.hidden_size / 4)) {
          threads /= 2U;
        }
        auto options = args.options;
        options.block_dim = {threads, 1U, 1U};
        return CudaKernelRuntime::CompileAndLaunch(
            tag, CudaKernelRuntime::MakeKernelSource<T>(
                     include, "omni_runtime::kernels::layernorm_kernel::LayernormVectorImpl"),
            options, static_cast<const T *>(args.input), static_cast<const T *>(args.weight),
            static_cast<const T *>(args.bias), args.rows, args.hidden_size, args.epsilon,
            static_cast<T *>(args.output));
      }
    }
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
                            .input = backend::cuda_internal::CudaAddress<void>(args.input),
                            .weight = backend::cuda_internal::CudaAddress<void>(args.weight),
                            .rows = args.rows,
                            .hidden_size = args.hidden_size,
                            .epsilon = args.epsilon,
                            .output = backend::cuda_internal::CudaAddress<void>(args.output),
                            .is_ggml_normalization = args.is_ggml_normalization,
                            .is_float_weight = args.is_float_weight});
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
    if constexpr (std::is_same_v<T, backend::BFloat16>) {
      if (!args.is_ggml_normalization && args.hidden_size >= 128 && args.hidden_size <= 5120 &&
          args.hidden_size % 4 == 0) {
        return CudaKernelRuntime::CompileAndLaunch(
            "backend_ResidualRmsNorm",
            CudaKernelRuntime::MakeKernelSource<T>(
                include, "omni_runtime::kernels::layernorm_kernel::BfloatRmsNormImpl"),
            {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
             .block_dim = {BfloatRmsNormThreads(args.rows, args.hidden_size), 1U, 1U},
             .shared_memory_bytes = 0},
            backend::cuda_internal::CudaAddress<const T>(args.input),
            backend::cuda_internal::CudaAddress<const T>(args.residual),
            backend::cuda_internal::CudaAddress<const void>(args.weight), args.rows,
            args.hidden_size, args.epsilon, args.is_float_weight,
            backend::cuda_internal::CudaAddress<T>(args.residual_output),
            backend::cuda_internal::CudaAddress<T>(args.output));
      }
    }
    return CudaKernelRuntime::CompileAndLaunch(
        "backend_ResidualRmsNorm",
        args.is_ggml_normalization ? CudaKernelRuntime::MakeKernelSource<T, 1>(include, kernel)
                                   : CudaKernelRuntime::MakeKernelSource<T, 0>(include, kernel),
        {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
         .block_dim = {threads, 1U, 1U},
         .shared_memory_bytes = 0},
        backend::cuda_internal::CudaAddress<const T>(args.input),
        backend::cuda_internal::CudaAddress<const T>(args.residual),
        backend::cuda_internal::CudaAddress<const void>(args.weight), args.rows, args.hidden_size,
        args.epsilon, backend::cuda_internal::CudaAddress<T>(args.residual_output),
        backend::cuda_internal::CudaAddress<T>(args.output), args.is_float_weight);
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
                              .input = backend::cuda_internal::CudaAddress<void>(args.input),
                              .weight = backend::cuda_internal::CudaAddress<void>(args.weight),
                              .bias = backend::cuda_internal::CudaAddress<void>(args.bias),
                              .rows = args.rows,
                              .hidden_size = args.hidden_size,
                              .epsilon = args.epsilon,
                              .output = backend::cuda_internal::CudaAddress<void>(args.output),
                              .is_ggml_normalization = args.is_ggml_normalization});
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
