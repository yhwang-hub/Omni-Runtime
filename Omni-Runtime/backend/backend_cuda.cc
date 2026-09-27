#include "Omni-Runtime/backend/backend_cuda.h"

#include <cuda_runtime.h>
#include <limits>
#include <filesystem>
#include <cstdlib>
#include "Omni-Runtime/kernels/add_kernel/add_kernel_runtime.h"
#include "Omni-Runtime/kernels/argmax_kernel/argmax_kernel_runtime.h"
#include "Omni-Runtime/kernels/embedding_kernel/embedding_kernel_runtime.h"
#include "Omni-Runtime/kernels/gqa_attention_kernel/gqa_attention_kernel_runtime.h"
#include "Omni-Runtime/kernels/flash_attention_v4_kernel/flash_attention_v4_kernel_runtime.h"
#include "Omni-Runtime/kernels/image_kernel/image_kernel_runtime.h"
#include "Omni-Runtime/kernels/kv_cache_kernel/kv_cache_kernel_runtime.h"
#include "Omni-Runtime/kernels/layernorm_kernel/layernorm_kernel_runtime.h"
#include "Omni-Runtime/kernels/mrope_kernel/mrope_kernel_runtime.h"
#include "Omni-Runtime/kernels/swiglu_kernel/swiglu_kernel_runtime.h"
#include "Omni-Runtime/kernels/vision_encoder_kernel/vision_encoder_kernel_runtime.h"
#include "Omni-Runtime/kernels/matmul_kernel/matmul_kernel_runtime.h"

#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace backend {
namespace {

cudaEvent_t NativeEvent(const void *event) {
  return reinterpret_cast<cudaEvent_t>(const_cast<void *>(event));
}

cudaStream_t NativeStream(const void *stream) {
  return reinterpret_cast<cudaStream_t>(const_cast<void *>(stream));
}

std::filesystem::path FindLibraryRoot() {
  const char *const root = std::getenv("OMNI_RUNTIME_ROOT");
  if (root != nullptr && root[0] != '\0') {
    return std::filesystem::path(root);
  }
  const char *const test_srcdir = std::getenv("TEST_SRCDIR");
  if (test_srcdir != nullptr && test_srcdir[0] != '\0') {
    return std::filesystem::path(test_srcdir) / "_main" / "Omni-Runtime";
  }
  std::error_code error;
  std::filesystem::path directory = std::filesystem::current_path(error);
  while (!error) {
    if (std::filesystem::is_directory(directory / "Omni-Runtime" / "kernels", error)) {
      return directory / "Omni-Runtime";
    }
    error.clear();
    const std::filesystem::path parent = directory.parent_path();
    if (parent == directory) {
      break;
    }
    directory = parent;
  }
  return std::filesystem::current_path();
}

template <typename T> bool LaunchCompute(const Embedding &args) {
  OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.hidden_size <= 0, false, ERROR,
                         "Invalid compute operation dimensions");
  return kernels::runtime::EmbeddingKernelRuntime<T>::CompileAndLaunch(
      "backend_Embedding",
      {.options = {.grid_dim = {static_cast<uint32_t>(args.token_count), 1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .indices = args.indices,
       .weight = args.weight,
       .token_count = args.token_count,
       .hidden_size = args.hidden_size,
       .output = args.output});
}

template <typename T> bool LaunchCompute(const RmsNorm &args) {
  OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.hidden_size <= 0, false, ERROR,
                         "Invalid compute operation dimensions");
  return kernels::runtime::RmsNormKernelRuntime<T>::CompileAndLaunch(
      "backend_RmsNorm", {.options = {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
                                      .block_dim = {256U, 1U, 1U},
                                      .shared_memory_bytes = 0},
                          .input = args.input,
                          .weight = args.weight,
                          .rows = args.rows,
                          .hidden_size = args.hidden_size,
                          .epsilon = args.epsilon,
                          .output = args.output});
}

template <typename T> bool LaunchCompute(const LayerNorm &args) {
  OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.hidden_size <= 0, false, ERROR,
                         "Invalid compute operation dimensions");
  return kernels::runtime::LayernormKernelRuntime<T>::CompileAndLaunch(
      "backend_LayerNorm", {.options = {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
                                        .block_dim = {256U, 1U, 1U},
                                        .shared_memory_bytes = 0},
                            .input = args.input,
                            .weight = args.weight,
                            .bias = args.bias,
                            .rows = args.rows,
                            .hidden_size = args.hidden_size,
                            .epsilon = args.epsilon,
                            .output = args.output});
}

template <typename T> bool LaunchCompute(const MultiAxisRope &args) {
  OMNI_RETURN_VAL_IF_LOG(args.tokens <= 0 || args.query_heads <= 0 || args.key_value_heads <= 0 ||
                             args.head_dim <= 0,
                         false, ERROR, "Invalid compute operation dimensions");
  return kernels::runtime::MropeKernelRuntime<T>::CompileAndLaunch(
      "backend_MultiAxisRope",
      {.options = {.grid_dim = {static_cast<uint32_t>(
                                    (uint64_t(args.tokens) * args.query_heads * args.head_dim +
                                     255U) /
                                    256U),
                                1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .query = args.query,
       .key = args.key,
       .position_ids = args.position_ids,
       .tokens = args.tokens,
       .query_heads = args.query_heads,
       .key_value_heads = args.key_value_heads,
       .head_dim = args.head_dim,
       .temporal_section = args.temporal_section,
       .height_section = args.height_section,
       .width_section = args.width_section,
       .frequency_base = args.frequency_base,
       .query_output = args.query_output,
       .key_output = args.key_output});
}

template <typename T> bool LaunchCompute(const StoreKvCache &args) {
  OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.key_value_size <= 0 ||
                             args.cache_capacity <= 0,
                         false, ERROR, "Invalid compute operation dimensions");
  return kernels::runtime::StoreKvCacheKernelRuntime<T>::CompileAndLaunch(
      "backend_StoreKvCache",
      {.options = {.grid_dim = {static_cast<uint32_t>(
                                    (uint64_t(args.token_count) * args.key_value_size + 255U) /
                                    256U),
                                1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .key = args.key,
       .value = args.value,
       .token_count = args.token_count,
       .key_value_size = args.key_value_size,
       .start_position = args.start_position,
       .cache_capacity = args.cache_capacity,
       .key_cache = args.key_cache,
       .value_cache = args.value_cache});
}

template <typename T> bool LaunchCompute(const Attention &args) {
  OMNI_RETURN_VAL_IF_LOG(args.query_token_count <= 0 || args.query_head_count <= 0 ||
                             args.key_value_head_count <= 0 || args.cached_token_count <= 0 ||
                             args.head_dim <= 0,
                         false, ERROR, "Invalid compute operation dimensions");
  const kernels::runtime::AttentionArgs parameters = {
      .options = {.grid_dim = {static_cast<uint32_t>(uint64_t(args.query_token_count) *
                                                     args.query_head_count),
                               1U, 1U},
                  .block_dim = {128U, 1U, 1U},
                  .shared_memory_bytes = 0},
      .query = args.query,
      .key_cache = args.key_cache,
      .value_cache = args.value_cache,
      .query_token_count = args.query_token_count,
      .query_head_count = args.query_head_count,
      .key_value_head_count = args.key_value_head_count,
      .cached_token_count = args.cached_token_count,
      .start_position = args.start_position,
      .head_dim = args.head_dim,
      .is_causal = args.is_causal,
      .output = args.output,
      .decode_partials = args.decode_partials};
  if constexpr (sizeof(T) == 2U) {
    if (args.query_token_count > 1 && args.head_dim <= 128 && kernel::jit::JitRuntime::HasSm120()) {
      return kernels::runtime::FlashAttentionV4KernelRuntime<T>::CompileAndLaunch(
          "backend_Attention_v4_sm120", parameters);
    }
  }
  return kernels::runtime::GqaAttentionKernelRuntime<T>::CompileAndLaunch("backend_Attention",
                                                                          parameters);
}

template <typename T> bool LaunchCompute(const Add &args) {
  return kernels::runtime::AddKernelRuntime<T>::CompileAndLaunch(
      "backend_Add",
      {.options = {.grid_dim = {static_cast<uint32_t>((args.element_count + 255U) / 256U), 1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .lhs = args.lhs,
       .rhs = args.rhs,
       .element_count = args.element_count,
       .output = args.output});
}

template <typename T> bool LaunchCompute(const SwiGLU &args) {
  return kernels::runtime::SwigluKernelRuntime<T>::CompileAndLaunch(
      "backend_SwiGLU",
      {.options = {.grid_dim = {static_cast<uint32_t>((args.element_count + 255U) / 256U), 1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .gate = args.gate,
       .up = args.up,
       .element_count = args.element_count,
       .output = args.output});
}

template <typename T> bool LaunchCompute(const Argmax &args) {
  OMNI_RETURN_VAL_IF_LOG(args.rows <= 0 || args.columns <= 0, false, ERROR,
                         "Invalid compute operation dimensions");
  return kernels::runtime::ArgmaxKernelRuntime<T>::CompileAndLaunch(
      "backend_Argmax", {.options = {.grid_dim = {static_cast<uint32_t>(args.rows), 1U, 1U},
                                     .block_dim = {256U, 1U, 1U},
                                     .shared_memory_bytes = 0},
                         .logits = args.logits,
                         .rows = args.rows,
                         .columns = args.columns,
                         .output = args.output});
}

template <typename T> bool LaunchCompute(const ImagePatchPreprocess &args) {
  OMNI_RETURN_VAL_IF_LOG(args.source_width_pixels <= 0 || args.source_height_pixels <= 0 ||
                             args.patch_size <= 0,
                         false, ERROR, "Invalid compute operation dimensions");
  return kernels::runtime::Qwen3VLPreprocessKernelRuntime<T>::CompileAndLaunch(
      "backend_ImagePatchPreprocess",
      {.options = {.grid_dim = {static_cast<uint32_t>((args.pixel_count + 255U) / 256U), 1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .pixel_count = args.pixel_count,
       .source = args.source,
       .source_width_pixels = args.source_width_pixels,
       .source_height_pixels = args.source_height_pixels,
       .patch_size = args.patch_size,
       .destination = args.destination,
       .input_width_pixels = args.input_width_pixels,
       .input_height_pixels = args.input_height_pixels});
}

template <typename T> bool LaunchCompute(const PairedPatchEmbedding &args) {
  OMNI_RETURN_VAL_IF_LOG(args.height <= 0 || args.width <= 0 || args.channels <= 0 ||
                             args.output_channels <= 0 || args.spatial_patch <= 0,
                         false, ERROR, "Invalid compute operation dimensions");
  return kernels::runtime::VisionPatchEmbedPairKernelRuntime<T>::CompileAndLaunch(
      "backend_PairedPatchEmbedding",
      {.options = {.grid_dim = {static_cast<uint32_t>(uint64_t(args.height / args.spatial_patch) *
                                                      (args.width / args.spatial_patch) *
                                                      args.output_channels),
                                1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .input = args.input,
       .first_weight = args.first_weight,
       .second_weight = args.second_weight,
       .bias = args.bias,
       .height = args.height,
       .width = args.width,
       .channels = args.channels,
       .output_channels = args.output_channels,
       .spatial_patch = args.spatial_patch,
       .output = args.output});
}

template <typename T> bool LaunchCompute(const GridPositionEmbedding &args) {
  OMNI_RETURN_VAL_IF_LOG(args.grid_height <= 0 || args.grid_width <= 0 ||
                             args.position_grid_size <= 0 || args.hidden_size <= 0 ||
                             args.merge_size <= 0,
                         false, ERROR, "Invalid compute operation dimensions");
  return kernels::runtime::VisionAddPositionKernelRuntime<T>::CompileAndLaunch(
      "backend_GridPositionEmbedding",
      {.options = {.grid_dim = {static_cast<uint32_t>((uint64_t(args.grid_height) *
                                                           args.grid_width * args.hidden_size +
                                                       255U) /
                                                      256U),
                                1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .input = args.input,
       .position_weight = args.position_weight,
       .grid_height = args.grid_height,
       .grid_width = args.grid_width,
       .position_grid_size = args.position_grid_size,
       .hidden_size = args.hidden_size,
       .merge_size = args.merge_size,
       .output = args.output});
}

template <typename T> bool LaunchCompute(const Bias &args) {
  OMNI_RETURN_VAL_IF_LOG(args.row_size <= 0, false, ERROR, "Invalid compute operation dimensions");
  return kernels::runtime::VisionBiasKernelRuntime<T>::CompileAndLaunch(
      "backend_Bias",
      {.options = {.grid_dim = {static_cast<uint32_t>((args.element_count + 255U) / 256U), 1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .input = args.input,
       .bias = args.bias,
       .element_count = args.element_count,
       .row_size = args.row_size,
       .output = args.output});
}

template <typename T> bool LaunchCompute(const BiasResidual &args) {
  OMNI_RETURN_VAL_IF_LOG(args.row_size <= 0, false, ERROR, "Invalid compute operation dimensions");
  return kernels::runtime::VisionBiasResidualKernelRuntime<T>::CompileAndLaunch(
      "backend_BiasResidual",
      {.options = {.grid_dim = {static_cast<uint32_t>((args.element_count + 255U) / 256U), 1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .input = args.input,
       .bias = args.bias,
       .residual = args.residual,
       .element_count = args.element_count,
       .row_size = args.row_size,
       .output = args.output});
}

template <typename T> bool LaunchCompute(const GeluBias &args) {
  OMNI_RETURN_VAL_IF_LOG(args.row_size <= 0, false, ERROR, "Invalid compute operation dimensions");
  return kernels::runtime::VisionGeluBiasKernelRuntime<T>::CompileAndLaunch(
      "backend_GeluBias",
      {.options = {.grid_dim = {static_cast<uint32_t>((args.element_count + 255U) / 256U), 1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .input = args.input,
       .bias = args.bias,
       .element_count = args.element_count,
       .row_size = args.row_size,
       .is_approximate = args.is_approximate,
       .output = args.output});
}

template <typename T> bool LaunchCompute(const GridQkvRope &args) {
  OMNI_RETURN_VAL_IF_LOG(args.token_count <= 0 || args.grid_height <= 0 || args.grid_width <= 0 ||
                             args.merge_size <= 0 || args.head_count <= 0 || args.head_size <= 0,
                         false, ERROR, "Invalid compute operation dimensions");
  return kernels::runtime::VisionQkvRopeKernelRuntime<T>::CompileAndLaunch(
      "backend_GridQkvRope",
      {.options = {.grid_dim = {static_cast<uint32_t>(
                                    (uint64_t(args.token_count) * args.head_count * args.head_size +
                                     255U) /
                                    256U),
                                1U, 1U},
                   .block_dim = {256U, 1U, 1U},
                   .shared_memory_bytes = 0},
       .qkv = args.qkv,
       .token_count = args.token_count,
       .grid_height = args.grid_height,
       .grid_width = args.grid_width,
       .merge_size = args.merge_size,
       .head_count = args.head_count,
       .head_size = args.head_size,
       .query = args.query,
       .key = args.key,
       .value = args.value});
}

template <typename Callable>
bool DispatchScalarType(const ScalarType type, const Callable &operation) {
  switch (type) {
  case ScalarType::kFloat32:
    return operation.template operator()<float>();
  case ScalarType::kFloat16:
    return operation.template operator()<kernels::Float16>();
  case ScalarType::kBFloat16:
    return operation.template operator()<kernels::BFloat16>();
  }
  LOG_ERROR("Unsupported compute scalar type");
  return false;
}

} // namespace

bool CudaBackend::Compute(const ScalarType type, const ComputeOperation &operation) {
  OMNI_RETURN_VAL_IF_LOG(!is_jit_initialized_, false, ERROR, "Compute backend is not initialized");
  const bool is_successful = DispatchScalarType(type, [&]<typename T>() {
    return std::visit([](const auto &args) { return LaunchCompute<T>(args); }, operation);
  });
  OMNI_RETURN_VAL_IF_LOG(!is_successful, false, ERROR, "CUDA compute operation failed");
  return true;
}

CudaBackend::~CudaBackend() {
  linear_plans_.clear();
  for (const cudaEvent_t event : linear_events_) {
    if (event != nullptr && cudaEventDestroy(event) != cudaSuccess) {
      LOG_ERROR("Failed to destroy a BLAS tuning event");
    }
  }
  if (linear_handle_ != nullptr && cublasLtDestroy(linear_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG_ERROR("Failed to destroy the BLAS linear handle");
  }
  if (linear_workspace_.data != nullptr && !Free(&linear_workspace_)) {
    LOG_ERROR("Failed to free the BLAS linear workspace");
  }
  if (blas_handle_ != nullptr && cublasDestroy(blas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG_ERROR("Failed to destroy the BLAS handle");
  }
}

CudaBackend::LinearPlan::~LinearPlan() {
  if (operation != nullptr && cublasLtMatmulDescDestroy(operation) != CUBLAS_STATUS_SUCCESS) {
    LOG_ERROR("Failed to destroy the BLAS linear descriptor");
  }
  for (const cublasLtMatrixLayout_t layout : layouts) {
    if (layout != nullptr && cublasLtMatrixLayoutDestroy(layout) != CUBLAS_STATUS_SUCCESS) {
      LOG_ERROR("Failed to destroy a BLAS matrix layout");
    }
  }
  if (preference != nullptr &&
      cublasLtMatmulPreferenceDestroy(preference) != CUBLAS_STATUS_SUCCESS) {
    LOG_ERROR("Failed to destroy the BLAS linear preference");
  }
}

bool CudaBackend::LinearPlan::Initialize(const cublasLtHandle_t handle, const cudaDataType_t type,
                                         const int32_t rows, const int32_t inner,
                                         const int32_t columns, const size_t workspace_bytes) {
  OMNI_RETURN_VAL_IF(handle == nullptr, false);
  const cublasOperation_t transpose = CUBLAS_OP_T;
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  const uint32_t alignment = 16U;
  const bool is_created =
      cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
                                     sizeof(transpose)) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&layouts[0], type, rows, inner, inner) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&layouts[1], type, columns, inner, inner) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&layouts[2], type, rows, columns, columns) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &workspace_bytes,
                                           sizeof(workspace_bytes)) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                           &reduction, sizeof(reduction)) == CUBLAS_STATUS_SUCCESS;
  OMNI_RETURN_VAL_IF_LOG(!is_created, false, ERROR, "Failed to configure the BLAS linear plan");
  for (const cublasLtMatrixLayout_t layout : layouts) {
    OMNI_RETURN_VAL_IF_LOG(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                                            &order,
                                                            sizeof(order)) != CUBLAS_STATUS_SUCCESS,
                           false, ERROR, "Failed to configure BLAS row-major layout");
  }
  for (const auto attribute :
       {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
    OMNI_RETURN_VAL_IF_LOG(cublasLtMatmulPreferenceSetAttribute(preference, attribute, &alignment,
                                                                sizeof(alignment)) !=
                               CUBLAS_STATUS_SUCCESS,
                           false, ERROR, "Failed to configure BLAS matrix alignment");
  }
  return true;
}

bool CudaBackend::LinearPlan::Tune(const cublasLtHandle_t handle, const void *const input,
                                   const void *const weight, const Allocation &workspace,
                                   const std::array<cudaEvent_t, 2U> &events, void *const output) {
  OMNI_RETURN_VAL_IF(handle == nullptr || input == nullptr || weight == nullptr ||
                         workspace.data == nullptr || events[0] == nullptr ||
                         events[1] == nullptr || output == nullptr,
                     false);
  std::array<cublasLtMatmulHeuristicResult_t, 8U> candidates = {};
  int32_t result_count = 0;
  OMNI_RETURN_VAL_IF_LOG(cublasLtMatmulAlgoGetHeuristic(handle, operation, layouts[0], layouts[1],
                                                        layouts[2], layouts[2], preference,
                                                        candidates.size(), candidates.data(),
                                                        &result_count) != CUBLAS_STATUS_SUCCESS ||
                             result_count == 0,
                         false, ERROR, "No FP32-accumulating BLAS linear algorithm");
  float best_time_ms = std::numeric_limits<float>::max();
  const float alpha = 1.0F;
  const float beta = 0.0F;
  for (int32_t index = 0; index < result_count; ++index) {
    const auto &candidate = candidates[index];
    OMNI_RETURN_VAL_IF_LOG(candidate.state != CUBLAS_STATUS_SUCCESS, false, ERROR,
                           "Invalid BLAS linear heuristic candidate");
    OMNI_RETURN_VAL_IF_LOG(cudaEventRecord(events[0]) != cudaSuccess, false, ERROR,
                           "Failed to record BLAS tuning start");
    for (int32_t iteration = 0; iteration < 3; ++iteration) {
      OMNI_RETURN_VAL_IF_LOG(cublasLtMatmul(handle, operation, &alpha, input, layouts[0], weight,
                                            layouts[1], &beta, output, layouts[2], output,
                                            layouts[2], &candidate.algo, workspace.data,
                                            workspace.size_bytes, nullptr) != CUBLAS_STATUS_SUCCESS,
                             false, ERROR, "BLAS tuning candidate failed");
    }
    OMNI_RETURN_VAL_IF_LOG(cudaEventRecord(events[1]) != cudaSuccess ||
                               cudaEventSynchronize(events[1]) != cudaSuccess,
                           false, ERROR, "Failed to synchronize BLAS tuning");
    float time_ms = 0.0F;
    OMNI_RETURN_VAL_IF_LOG(cudaEventElapsedTime(&time_ms, events[0], events[1]) != cudaSuccess,
                           false, ERROR, "Failed to measure BLAS tuning time");
    if (time_ms < best_time_ms) {
      best_time_ms = time_ms;
      algorithm = candidate.algo;
    }
  }
  return true;
}

bool CudaBackend::InitializeCompute() {
  const std::lock_guard<std::mutex> lock(compute_mutex_);
  if (!is_jit_initialized_) {
    OMNI_RETURN_VAL_IF_LOG(!kernel::jit::JitRuntime::Init(FindLibraryRoot()), false, ERROR,
                           "Failed to initialize CUDA compute kernels");
    is_jit_initialized_ = true;
  }
  if (blas_handle_ == nullptr) {
    OMNI_RETURN_VAL_IF_LOG(cublasCreate(&blas_handle_) != CUBLAS_STATUS_SUCCESS, false, ERROR,
                           "Failed to initialize BLAS");
  }
  OMNI_RETURN_VAL_IF_LOG(
      cublasSetMathMode(blas_handle_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION) !=
          CUBLAS_STATUS_SUCCESS,
      false, ERROR, "Failed to configure FP32 BLAS accumulation");
  if (linear_handle_ == nullptr) {
    OMNI_RETURN_VAL_IF_LOG(cublasLtCreate(&linear_handle_) != CUBLAS_STATUS_SUCCESS, false, ERROR,
                           "Failed to initialize BLAS linear compute");
  }
  if (linear_workspace_.data == nullptr) {
    OMNI_RETURN_VAL_IF_LOG(!Allocate(MemoryKind::kDevice, 32U * 1024U * 1024U, &linear_workspace_),
                           false, ERROR, "Failed to allocate BLAS linear workspace");
  }
  for (cudaEvent_t &event : linear_events_) {
    if (event == nullptr) {
      OMNI_RETURN_VAL_IF_LOG(cudaEventCreate(&event) != cudaSuccess, false, ERROR,
                             "Failed to create a BLAS tuning event");
    }
  }
  return true;
}

size_t CudaBackend::compute_workspace_bytes() const {
  return linear_workspace_.size_bytes;
}

bool CudaBackend::LinearMatmul(const ScalarType type, const void *const input,
                               const void *const weight, const int32_t rows, const int32_t inner,
                               const int32_t columns, void *const output) {
  OMNI_RETURN_VAL_IF_LOG(input == nullptr || weight == nullptr || output == nullptr ||
                             blas_handle_ == nullptr || rows <= 0 || inner <= 0 || columns <= 0 ||
                             (type != ScalarType::kFloat32 && type != ScalarType::kFloat16 &&
                              type != ScalarType::kBFloat16),
                         false, ERROR, "Invalid BLAS linear multiplication inputs");
  if (rows == 1) {
    return DispatchScalarType(type, [&]<typename T>() {
      return kernels::runtime::LinearMatmulKernelRuntime<T>::CompileAndLaunch("backend_gemv",
                                                                              {.options = {},
                                                                               .lhs = input,
                                                                               .rhs = weight,
                                                                               .rows = rows,
                                                                               .inner = inner,
                                                                               .columns = columns,
                                                                               .output = output});
    });
  }
  const std::lock_guard<std::mutex> lock(compute_mutex_);
  const cudaDataType_t data_type = type == ScalarType::kFloat32   ? CUDA_R_32F
                                   : type == ScalarType::kFloat16 ? CUDA_R_16F
                                                                  : CUDA_R_16BF;
  const float alpha = 1.0F;
  const float beta = 0.0F;
  if (type != ScalarType::kFloat32 && rows > 1 && inner % 8 == 0 && columns % 8 == 0 &&
      reinterpret_cast<uintptr_t>(input) % 16U == 0U &&
      reinterpret_cast<uintptr_t>(weight) % 16U == 0U &&
      reinterpret_cast<uintptr_t>(output) % 16U == 0U) {
    const std::array<int32_t, 4U> key = {static_cast<int32_t>(type), rows, inner, columns};
    auto iterator = linear_plans_.find(key);
    if (iterator == linear_plans_.end()) {
      auto plan = std::make_unique<LinearPlan>();
      OMNI_RETURN_VAL_IF(!plan->Initialize(linear_handle_, data_type, rows, inner, columns,
                                           linear_workspace_.size_bytes),
                         false);
      OMNI_RETURN_VAL_IF(
          !plan->Tune(linear_handle_, input, weight, linear_workspace_, linear_events_, output),
          false);
      iterator = linear_plans_.emplace(key, std::move(plan)).first;
    }
    const LinearPlan &plan = *iterator->second;
    OMNI_RETURN_VAL_IF_LOG(cublasLtMatmul(linear_handle_, plan.operation, &alpha, input,
                                          plan.layouts[0], weight, plan.layouts[1], &beta, output,
                                          plan.layouts[2], output, plan.layouts[2], &plan.algorithm,
                                          linear_workspace_.data, linear_workspace_.size_bytes,
                                          nullptr) != CUBLAS_STATUS_SUCCESS,
                           false, ERROR, "BLAS linear plan execution failed");
    return true;
  }
  const cublasStatus_t status =
      cublasGemmEx(blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, inner, &alpha, weight,
                   data_type, inner, input, data_type, inner, &beta, output, data_type, columns,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  OMNI_RETURN_VAL_IF_LOG(status != CUBLAS_STATUS_SUCCESS, false, ERROR,
                         "BLAS linear multiplication failed: " << static_cast<int>(status));
  return true;
}

bool CudaBackend::IsAvailable() const {
  int device_index = 0;
  return cudaGetDevice(&device_index) == cudaSuccess;
}

BackendKind CudaBackend::kind() const {
  return BackendKind::kCuda;
}

bool CudaBackend::IsMemoryKindSupported(const MemoryKind memory_kind) const {
  return memory_kind == MemoryKind::kPinnedHost || memory_kind == MemoryKind::kManaged ||
         memory_kind == MemoryKind::kDevice;
}

bool CudaBackend::Allocate(const MemoryKind memory_kind, const std::size_t size_bytes,
                           Allocation *const allocation) {
  OMNI_RETURN_VAL_IF(allocation == nullptr || size_bytes == 0U, false);
  OMNI_RETURN_VAL_IF(allocation->data != nullptr, false);

  void *data = nullptr;
  cudaError_t status = cudaSuccess;
  if (memory_kind == MemoryKind::kPinnedHost) {
    status = cudaMallocHost(&data, size_bytes);
  } else if (memory_kind == MemoryKind::kManaged) {
    status = cudaMallocManaged(&data, size_bytes, cudaMemAttachGlobal);
  } else if (memory_kind == MemoryKind::kDevice) {
    status = cudaMalloc(&data, size_bytes);
  } else {
    return false;
  }
  OMNI_RETURN_VAL_IF(status != cudaSuccess || data == nullptr, false);

  allocation->data = data;
  allocation->size_bytes = size_bytes;
  allocation->memory_kind = memory_kind;
  return true;
}

bool CudaBackend::Free(Allocation *const allocation) {
  OMNI_RETURN_VAL_IF(allocation == nullptr, false);
  OMNI_RETURN_VAL_IF(allocation->data == nullptr, true);
  const cudaError_t status = cudaFree(allocation->data);
  allocation->data = nullptr;
  allocation->size_bytes = 0U;
  return status == cudaSuccess;
}

bool CudaBackend::Fill(const Allocation &allocation, const uint8_t value, const void *stream) {
  OMNI_RETURN_VAL_IF(allocation.data == nullptr, false);
  return Submit(stream, [&](cudaStream_t native_stream) {
    return cudaMemsetAsync(allocation.data, value, allocation.size_bytes, native_stream);
  });
}

bool CudaBackend::CopyHostToDevice(const void *const source, const std::size_t size_bytes,
                                   const Allocation &destination, const void *stream) {
  OMNI_RETURN_VAL_IF(source == nullptr || destination.data == nullptr, false);
  return Submit(stream, [&](cudaStream_t native_stream) {
    return cudaMemcpyAsync(destination.data, source, size_bytes, cudaMemcpyHostToDevice,
                           native_stream);
  });
}

bool CudaBackend::CopyDeviceToHost(const Allocation &source, const std::size_t size_bytes,
                                   void *const destination, const void *stream) {
  OMNI_RETURN_VAL_IF(destination == nullptr || source.data == nullptr, false);
  return Submit(stream, [&](cudaStream_t native_stream) {
    return cudaMemcpyAsync(destination, source.data, size_bytes, cudaMemcpyDeviceToHost,
                           native_stream);
  });
}

bool CudaBackend::CopyDeviceToDevice(const Allocation &source, const Allocation &destination,
                                     const std::size_t size_bytes, const void *stream) {
  OMNI_RETURN_VAL_IF(source.data == nullptr || destination.data == nullptr, false);
  return Submit(stream, [&](cudaStream_t native_stream) {
    return cudaMemcpyAsync(destination.data, source.data, size_bytes, cudaMemcpyDeviceToDevice,
                           native_stream);
  });
}

bool CudaBackend::Submit(const void *stream,
                         const std::function<cudaError_t(cudaStream_t)> &operation) {
  cudaStream_t native_stream = NativeStream(stream);
  if (operation(native_stream) != cudaSuccess) {
    return false;
  }
  return stream != nullptr || cudaStreamSynchronize(native_stream) == cudaSuccess;
}

bool CudaBackend::CreateStream(const uint32_t flags, StreamHandle *const stream_handle) {
  OMNI_RETURN_VAL_IF(stream_handle == nullptr || stream_handle->native_handle != nullptr, false);
  cudaStream_t native_stream = nullptr;
  const cudaError_t status = cudaStreamCreateWithFlags(&native_stream, flags);
  OMNI_RETURN_VAL_IF(status != cudaSuccess || native_stream == nullptr, false);
  stream_handle->native_handle = native_stream;
  stream_handle->backend_kind = BackendKind::kCuda;
  return true;
}

bool CudaBackend::SynchronizeStream(const void *stream) {
  return cudaStreamSynchronize(NativeStream(stream)) == cudaSuccess;
}

bool CudaBackend::QueryStream(const void *stream, bool *const is_ready) {
  OMNI_RETURN_VAL_IF(stream == nullptr || is_ready == nullptr, false);
  const cudaError_t status = cudaStreamQuery(NativeStream(stream));
  *is_ready = status == cudaSuccess;
  return status == cudaSuccess || status == cudaErrorNotReady;
}

bool CudaBackend::WaitEvent(const void *stream, const void *event, const uint32_t flags) {
  OMNI_RETURN_VAL_IF(stream == nullptr || event == nullptr, false);
  return cudaStreamWaitEvent(NativeStream(stream), NativeEvent(event), flags) == cudaSuccess;
}

bool CudaBackend::CreateEvent(const uint32_t flags, EventHandle *const event_handle) {
  OMNI_RETURN_VAL_IF(event_handle == nullptr || event_handle->native_handle != nullptr, false);
  cudaEvent_t native_event = nullptr;
  const cudaError_t status = cudaEventCreateWithFlags(&native_event, flags);
  OMNI_RETURN_VAL_IF(status != cudaSuccess || native_event == nullptr, false);
  event_handle->native_handle = native_event;
  event_handle->backend_kind = BackendKind::kCuda;
  return true;
}

bool CudaBackend::DestroyEvent(void *event) {
  if (event == nullptr) {
    return true;
  }
  return cudaEventDestroy(NativeEvent(event)) == cudaSuccess;
}

bool CudaBackend::RecordEvent(const void *stream, EventHandle *const event_handle) {
  OMNI_RETURN_VAL_IF(event_handle == nullptr || event_handle->native_handle == nullptr, false);
  return cudaEventRecord(NativeEvent(event_handle->native_handle), NativeStream(stream)) ==
         cudaSuccess;
}

bool CudaBackend::ElapsedTimeMs(const void *const start, const void *const end,
                                double *const elapsed_ms) {
  OMNI_RETURN_VAL_IF(start == nullptr || end == nullptr || elapsed_ms == nullptr, false);
  float milliseconds = 0.0F;
  OMNI_RETURN_VAL_IF_LOG(
      cudaEventElapsedTime(&milliseconds, NativeEvent(start), NativeEvent(end)) != cudaSuccess,
      false, ERROR, "CUDA event timing failed");
  *elapsed_ms = milliseconds;
  return true;
}

bool CudaBackend::SynchronizeEvent(const void *event) {
  OMNI_RETURN_VAL_IF(event == nullptr, false);
  return cudaEventSynchronize(NativeEvent(event)) == cudaSuccess;
}

bool CudaBackend::QueryEvent(const void *event, bool *const is_ready) {
  OMNI_RETURN_VAL_IF(event == nullptr || is_ready == nullptr, false);
  const cudaError_t status = cudaEventQuery(NativeEvent(event));
  *is_ready = status == cudaSuccess;
  return status == cudaSuccess || status == cudaErrorNotReady;
}

bool CudaBackend::DestroyStream(void *stream) {
  if (stream == nullptr) {
    return true;
  }
  return cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess;
}

} // namespace backend
} // namespace omni_runtime
