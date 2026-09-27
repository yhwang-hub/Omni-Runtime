#include "Omni-Runtime/backend/backend_cuda.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>

#include <cuda_runtime.h>

#include "Omni-Runtime/backend/backend_cuda_utils.h"
#include "Omni-Runtime/backend/runtime_path.h"
#include "Omni-Runtime/kernels/compute_runtime.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace backend {
namespace {

using cuda_internal::AllocationAddress;
using cuda_internal::CudaAddress;
using cuda_internal::NativeEvent;
using cuda_internal::NativeStream;

constexpr int32_t kParallelProjectionTuneMaxRows = 1024;

bool HasLinearPlan(const ScalarType type, const Linear &operation) {
  return ((type != ScalarType::kFloat32 && type != ScalarType::kBFloat16) ||
          operation.bias.IsValid()) &&
         (operation.rows > 1 || operation.bias.IsValid()) && operation.inner % 8 == 0 &&
         operation.columns % 8 == 0 &&
         reinterpret_cast<uintptr_t>(CudaAddress<void>(operation.input)) % 16U == 0U &&
         reinterpret_cast<uintptr_t>(CudaAddress<void>(operation.weight)) % 16U == 0U &&
         reinterpret_cast<uintptr_t>(CudaAddress<void>(operation.output)) % 16U == 0U;
}

template <typename Callable>
bool DispatchScalarType(const ScalarType type, const Callable &operation) {
  switch (type) {
  case ScalarType::kFloat32:
    return operation.template operator()<float>();
  case ScalarType::kFloat16:
    return operation.template operator()<Float16>();
  case ScalarType::kBFloat16:
    return operation.template operator()<BFloat16>();
  }
  LOG_ERROR("Unsupported compute scalar type");
  return false;
}

} // namespace

struct CudaBackend::ParallelRuntime final {
  static constexpr size_t kWorkspaceBytes = 32U * 1024U * 1024U;
  std::array<cudaStream_t, 2U> streams = {};
  std::array<cudaEvent_t, 2U> completed = {};
  std::array<cublasHandle_t, 2U> handles = {};
  std::array<Allocation, 2U> workspaces;
  cudaEvent_t ready = nullptr;

  ~ParallelRuntime() {
    if (!Drain()) {
      LOG_ERROR("Failed to drain parallel CUDA projections during destruction");
    }
    for (size_t lane = 0U; lane < streams.size(); ++lane) {
      if (handles[lane] != nullptr && cublasDestroy(handles[lane]) != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("Failed to destroy parallel BLAS handle");
      }
      if (completed[lane] != nullptr && cudaEventDestroy(completed[lane]) != cudaSuccess) {
        LOG_ERROR("Failed to destroy parallel completion event");
      }
      if (workspaces[lane].data != nullptr && cudaFree(workspaces[lane].data) != cudaSuccess) {
        LOG_ERROR("Failed to free parallel BLAS workspace");
      }
      if (streams[lane] != nullptr && cudaStreamDestroy(streams[lane]) != cudaSuccess) {
        LOG_ERROR("Failed to destroy parallel projection stream");
      }
    }
    if (ready != nullptr && cudaEventDestroy(ready) != cudaSuccess) {
      LOG_ERROR("Failed to destroy parallel readiness event");
    }
  }

  bool Initialize() {
    OMNI_RETURN_VAL_IF_LOG(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming) != cudaSuccess,
                           false, ERROR, "Failed to create parallel readiness event");
    for (size_t lane = 0U; lane < streams.size(); ++lane) {
      OMNI_RETURN_VAL_IF_LOG(
          cudaStreamCreateWithFlags(&streams[lane], cudaStreamNonBlocking) != cudaSuccess ||
              cudaEventCreateWithFlags(&completed[lane], cudaEventDisableTiming) != cudaSuccess ||
              cudaMalloc(&workspaces[lane].data, kWorkspaceBytes) != cudaSuccess ||
              cublasCreate(&handles[lane]) != CUBLAS_STATUS_SUCCESS ||
              cublasSetStream(handles[lane], streams[lane]) != CUBLAS_STATUS_SUCCESS ||
              cublasSetMathMode(handles[lane], CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION) !=
                  CUBLAS_STATUS_SUCCESS,
          false, ERROR, "Failed to initialize parallel BLAS lane");
      workspaces[lane].size_bytes = kWorkspaceBytes;
    }
    return true;
  }

  bool Drain() const {
    bool is_successful = true;
    for (const cudaStream_t stream : streams) {
      if (stream != nullptr && cudaStreamSynchronize(stream) != cudaSuccess) {
        is_successful = false;
      }
    }
    return is_successful;
  }
};

CudaBackend::CudaBackend() = default;

struct CudaBackend::MatmulWorkspace final {
  std::array<Allocation, 2U> buffers = {};

  ~MatmulWorkspace() {
    for (const Allocation &buffer : buffers) {
      if (buffer.data != nullptr && cudaFree(buffer.data) != cudaSuccess) {
        LOG_ERROR("Failed to free a BLAS workspace");
      }
    }
  }
};

bool CudaBackend::Compute(const ScalarType type, const ComputeOperation &operation) {
  OMNI_RETURN_VAL_IF_LOG(!is_jit_initialized_, false, ERROR, "Compute backend is not initialized");
  const bool is_successful = DispatchScalarType(type, [&]<typename T>() {
    return std::visit(
        [this, type](const auto &args) {
          if constexpr (std::is_same_v<std::decay_t<decltype(args)>, Linear>) {
            OMNI_RETURN_VAL_IF_LOG(!args.bias.IsValid() &&
                                       (args.bias.data != nullptr || args.bias.offset_bytes != 0U ||
                                        args.bias.size_bytes != 0U),
                                   false, ERROR, "Invalid optional linear bias");
            if (args.bias.IsValid()) {
              return BiasedLinear(type, args);
            }
            return LinearMatmul(type, args.input, args.weight, args.rows, args.inner, args.columns,
                                args.output);
          } else if constexpr (std::is_same_v<std::decay_t<decltype(args)>, ParallelLinear>) {
            OMNI_RETURN_VAL_IF(!ValidateParallelLinear(type, kind(), args), false);
            bool is_enabled = false;
            OMNI_RETURN_VAL_IF(!GetParallelExecutionEnabled(&is_enabled), false);
            if (!is_enabled ||
                (args.projections.front().rows == 1 &&
                 !kernels::runtime::ParallelGemvKernelRuntime<T>::CanCombine(args))) {
              for (const Linear &projection : args.projections) {
                OMNI_RETURN_VAL_IF(!Compute(type, ComputeOperation(projection)), false);
              }
              return true;
            }
            if (args.projections.front().rows == 1) {
              return kernels::runtime::ParallelGemvKernelRuntime<T>::CompileAndLaunch(args);
            }
            return ParallelMatmul(type, args);
          } else if constexpr (std::is_same_v<std::decay_t<decltype(args)>, QuantizedLinear>) {
            if (type == ScalarType::kFloat32 && args.rows > 8 && !args.is_packed_nvfp4 &&
                ((args.weight_type == 40 && args.activation_scales.IsValid() &&
                  args.activation_block_size == 16) ||
                 (args.weight_type == 30 && !args.activation_scales.IsValid()))) {
              return QuantizedMatmul(args);
            }
            return kernels::runtime::CudaOperator<T, QuantizedLinear>::Launch(args);
          } else if constexpr (std::is_same_v<std::decay_t<decltype(args)>, Fp8Linear>) {
            return Fp8Matmul(type, args);
          } else if constexpr (std::is_same_v<std::decay_t<decltype(args)>, GatedDeltaNet>) {
            OMNI_RETURN_VAL_IF_LOG(args.is_chunked && type != ScalarType::kBFloat16, false, ERROR,
                                   "Chunked delta net requires BF16 storage");
            if (args.is_chunked && args.token_count > 1) {
              return ChunkDeltaNet(args);
            }
            return kernels::runtime::CudaOperator<T, GatedDeltaNet>::Launch(args);
          } else {
            return kernels::runtime::CudaOperator<T, std::decay_t<decltype(args)>>::Launch(args);
          }
        },
        operation);
  });
  OMNI_RETURN_VAL_IF_LOG(!is_successful, false, ERROR, "CUDA compute operation failed");
  return true;
}

class CudaBackend::CudaExecutionPlan final : public ExecutionPlan {
public:
  ~CudaExecutionPlan() override {
    if (stream_ != nullptr && cudaStreamSynchronize(stream_) != cudaSuccess) {
      LOG_ERROR("Failed to synchronize an execution plan during destruction");
    }
    if (executable_ != nullptr && cudaGraphExecDestroy(executable_) != cudaSuccess) {
      LOG_ERROR("Failed to destroy an execution graph");
    }
    if (graph_ != nullptr && cudaGraphDestroy(graph_) != cudaSuccess) {
      LOG_ERROR("Failed to destroy a captured graph");
    }
    for (const cudaEvent_t event : events_) {
      if (event != nullptr && cudaEventDestroy(event) != cudaSuccess) {
        LOG_ERROR("Failed to destroy an execution timing event");
      }
    }
    if (inputs_ready_ != nullptr && cudaEventDestroy(inputs_ready_) != cudaSuccess) {
      LOG_ERROR("Failed to destroy an execution input event");
    }
    if (result_ != nullptr && cudaFreeHost(result_) != cudaSuccess) {
      LOG_ERROR("Failed to free execution readback storage");
    }
    if (stream_ != nullptr && cudaStreamDestroy(stream_) != cudaSuccess) {
      LOG_ERROR("Failed to destroy an execution stream");
    }
  }

  bool Initialize(CudaBackend *const backend, const ScalarType type,
                  const std::vector<ComputeOperation> &operations, const Allocation &readback) {
    OMNI_RETURN_VAL_IF(backend == nullptr, false);
    readback_ = readback;
    OMNI_RETURN_VAL_IF_LOG(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaMallocHost(&result_, readback.size_bytes) != cudaSuccess ||
            cudaEventCreateWithFlags(&inputs_ready_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreate(&events_[0]) != cudaSuccess ||
            cudaEventCreate(&events_[1]) != cudaSuccess,
        false, ERROR, "Failed to allocate execution plan resources");
    bool is_parallel_enabled = false;
    OMNI_RETURN_VAL_IF(!GetParallelExecutionEnabled(&is_parallel_enabled), false);
    std::vector<size_t> widths(operations.size(), 1U);
    if (is_parallel_enabled && kernel::jit::JitRuntime::HasSm120()) {
      for (size_t index = 0U; index < operations.size();) {
        size_t next = index + 1U;
        while (next < operations.size()) {
          bool is_independent = true;
          for (size_t previous = index; previous < next; ++previous) {
            is_independent = is_independent && CanFork(operations[previous], operations[next]);
          }
          if (!is_independent) {
            break;
          }
          ++next;
        }
        widths[index] = next - index;
        index = next;
      }
    }
    const auto launch = [&] {
      for (size_t index = 0U; index < operations.size(); ++index) {
        if (widths[index] == 1U || kernel::jit::JitRuntime::is_preparing()) {
          OMNI_RETURN_VAL_IF(!backend->Compute(type, operations[index]), false);
          continue;
        }
        std::vector<cudaGraphNode_t> before;
        std::vector<cudaGraphNode_t> completed;
        std::vector<cudaGraphNode_t> branch_end;
        OMNI_RETURN_VAL_IF(!GetCaptureDependencies(&before), false);
        const size_t stop = index + widths[index];
        for (size_t branch = index; branch < stop; ++branch) {
          OMNI_RETURN_VAL_IF(!SetCaptureDependencies(before), false);
          OMNI_RETURN_VAL_IF(!backend->Compute(type, operations[branch]) ||
                                 !GetCaptureDependencies(&branch_end),
                             false);
          completed.insert(completed.end(), branch_end.begin(), branch_end.end());
        }
        OMNI_RETURN_VAL_IF(!SetCaptureDependencies(completed), false);
        index = stop - 1U;
      }
      return true;
    };
    {
      const kernel::jit::JitRuntime::LaunchScope scope(stream_, true, &kernels_);
      OMNI_RETURN_VAL_IF(!launch(), false);
    }
    OMNI_RETURN_VAL_IF_LOG(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) !=
                               cudaSuccess,
                           false, ERROR, "Failed to begin execution graph capture");
    bool is_captured = false;
    {
      const kernel::jit::JitRuntime::LaunchScope scope(stream_, false, &kernels_);
      is_captured = launch();
    }
    const cudaError_t capture_status = cudaStreamEndCapture(stream_, &graph_);
    OMNI_RETURN_VAL_IF_LOG(!is_captured || capture_status != cudaSuccess, false, ERROR,
                           "Failed to capture execution graph");
    OMNI_RETURN_VAL_IF_LOG(cudaGraphInstantiate(&executable_, graph_, 0) != cudaSuccess ||
                               cudaGraphUpload(executable_, stream_) != cudaSuccess ||
                               cudaStreamSynchronize(stream_) != cudaSuccess,
                           false, ERROR, "Failed to instantiate or upload execution graph");
    return true;
  }

  bool Run(const size_t result_bytes, void *const result, double *const elapsed_ms) override {
    OMNI_RETURN_VAL_IF_LOG(result == nullptr || elapsed_ms == nullptr || executable_ == nullptr ||
                               result_bytes != readback_.size_bytes || !is_usable_,
                           false, ERROR, "Invalid execution plan outputs or state");
    if (!Execute(result, elapsed_ms)) {
      is_usable_ = false;
      if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        LOG_ERROR("Failed to drain an invalid CUDA execution plan");
      }
      return false;
    }
    return true;
  }

private:
  bool SetCaptureDependencies(const std::span<cudaGraphNode_t> &nodes) const {
#if CUDART_VERSION >= 13000
    const cudaError_t status = cudaStreamUpdateCaptureDependencies(
        stream_, nodes.data(), nullptr, nodes.size(), cudaStreamSetCaptureDependencies);
#else
    const cudaError_t status = cudaStreamUpdateCaptureDependencies(
        stream_, nodes.data(), nodes.size(), cudaStreamSetCaptureDependencies);
#endif
    OMNI_RETURN_VAL_IF_LOG(status != cudaSuccess, false, ERROR,
                           "Failed to update captured projection dependencies");
    return true;
  }

  bool GetCaptureDependencies(std::vector<cudaGraphNode_t> *const nodes) const {
    OMNI_RETURN_VAL_IF(nodes == nullptr, false);
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    const cudaGraphNode_t *dependencies = nullptr;
    size_t count = 0U;
#if CUDART_VERSION >= 13000
    const cudaError_t result = cudaStreamGetCaptureInfo(stream_, &status, nullptr, nullptr,
                                                        &dependencies, nullptr, &count);
#else
    const cudaError_t result =
        cudaStreamGetCaptureInfo_v2(stream_, &status, nullptr, nullptr, &dependencies, &count);
#endif
    OMNI_RETURN_VAL_IF_LOG(result != cudaSuccess || status != cudaStreamCaptureStatusActive ||
                               (count != 0U && dependencies == nullptr),
                           false, ERROR, "Failed to inspect captured projection dependencies");
    nodes->clear();
    if (count != 0U) {
      // The capture owns this array and can replace it when its frontier changes.
      nodes->assign(dependencies, dependencies + count);
    }
    return true;
  }

  static bool CanFork(const ComputeOperation &first, const ComputeOperation &second) {
    const auto *const left = std::get_if<QuantizedLinear>(&first);
    const auto *const right = std::get_if<QuantizedLinear>(&second);
    if (left == nullptr || right == nullptr || left->rows != 1 || right->rows != 1 ||
        !left->is_packed_q8 || !right->is_packed_q8 || left->activation_block_size != 32 ||
        right->activation_block_size != 32 || left->inner != right->inner ||
        left->input.data != right->input.data ||
        left->input.offset_bytes != right->input.offset_bytes) {
      return false;
    }
    const DeviceView left_views[] = {left->input, left->weight, left->activation_scales,
                                     left->output};
    const DeviceView right_views[] = {right->input, right->weight, right->activation_scales,
                                      right->output};
    for (const DeviceView &view : left_views) {
      if (HasOverlap(right->output, view)) {
        return false;
      }
    }
    for (const DeviceView &view : right_views) {
      if (HasOverlap(left->output, view)) {
        return false;
      }
    }
    return true;
  }

  bool Execute(void *const result, double *const elapsed_ms) {
    OMNI_RETURN_VAL_IF_LOG(cudaEventRecord(inputs_ready_, nullptr) != cudaSuccess ||
                               cudaStreamWaitEvent(stream_, inputs_ready_, 0U) != cudaSuccess ||
                               cudaEventRecord(events_[0], stream_) != cudaSuccess ||
                               cudaGraphLaunch(executable_, stream_) != cudaSuccess ||
                               cudaEventRecord(events_[1], stream_) != cudaSuccess ||
                               cudaMemcpyAsync(result_, AllocationAddress(readback_),
                                               readback_.size_bytes, cudaMemcpyDeviceToHost,
                                               stream_) != cudaSuccess ||
                               cudaStreamSynchronize(stream_) != cudaSuccess,
                           false, ERROR, "Execution graph or result readback failed");
    float milliseconds = 0.0F;
    OMNI_RETURN_VAL_IF_LOG(cudaEventElapsedTime(&milliseconds, events_[0], events_[1]) !=
                               cudaSuccess,
                           false, ERROR, "Execution graph event timing failed");
    std::memcpy(result, result_, readback_.size_bytes);
    *elapsed_ms = milliseconds;
    return true;
  }

  bool is_usable_ = true;
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
  std::array<cudaEvent_t, 2U> events_ = {};
  cudaEvent_t inputs_ready_ = nullptr;
  Allocation readback_;
  void *result_ = nullptr;
  std::vector<std::shared_ptr<kernel::jit::CudaKernel>> kernels_;
};

bool CudaBackend::CreateExecutionPlan(const ScalarType type,
                                      const std::vector<ComputeOperation> &operations,
                                      const Allocation &readback,
                                      std::unique_ptr<ExecutionPlan> *const plan) {
  OMNI_RETURN_VAL_IF_LOG(!is_jit_initialized_ || plan == nullptr || operations.empty() ||
                             readback.data == nullptr || readback.size_bytes == 0U ||
                             (readback.memory_kind != MemoryKind::kDevice &&
                              readback.memory_kind != MemoryKind::kDeviceView),
                         false, ERROR, "Invalid execution plan inputs");
  for (const ComputeOperation &operation : operations) {
    const Linear *const linear = std::get_if<Linear>(&operation);
    OMNI_RETURN_VAL_IF_LOG(linear != nullptr && linear->rows != 1 && type != ScalarType::kFloat32 &&
                               type != ScalarType::kBFloat16,
                           false, ERROR,
                           "Prepared multirow linear operations require FP32 storage");
    const auto *const group = std::get_if<ParallelLinear>(&operation);
    OMNI_RETURN_VAL_IF_LOG(group != nullptr && (!ValidateParallelLinear(type, kind(), *group) ||
                                                group->projections.front().rows != 1),
                           false, ERROR, "Prepared parallel projections require one row");
  }
  auto prepared = std::make_unique<CudaExecutionPlan>();
  OMNI_RETURN_VAL_IF(!prepared->Initialize(this, type, operations, readback), false);
  *plan = std::move(prepared);
  return true;
}

CudaBackend::~CudaBackend() {
  parallel_runtime_.reset();
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
  quantized_workspaces_.clear();
  fp8_workspaces_.clear();
  biased_workspaces_.clear();
  delta_workspaces_.clear();
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
                                         const int32_t columns, const size_t workspace_bytes,
                                         const bool has_bias, const bool is_float_bias) {
  OMNI_RETURN_VAL_IF(handle == nullptr, false);
  const cublasOperation_t transpose = CUBLAS_OP_T;
  const cublasLtOrder_t order = has_bias ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
  const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  const uint32_t alignment = 16U;
  const bool is_created =
      cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulDescSetAttribute(
          operation, has_bias ? CUBLASLT_MATMUL_DESC_TRANSA : CUBLASLT_MATMUL_DESC_TRANSB,
          &transpose, sizeof(transpose)) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&layouts[0], type, has_bias ? inner : rows,
                                 has_bias ? columns : inner, inner) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&layouts[1], type, has_bias ? inner : columns,
                                 has_bias ? rows : inner, inner) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&layouts[2], has_bias && is_float_bias ? CUDA_R_32F : type,
                                 has_bias ? columns : rows, has_bias ? rows : columns,
                                 columns) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &workspace_bytes,
                                           sizeof(workspace_bytes)) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                           &reduction, sizeof(reduction)) == CUBLAS_STATUS_SUCCESS;
  OMNI_RETURN_VAL_IF_LOG(!is_created, false, ERROR, "Failed to configure the BLAS linear plan");
  if (has_bias) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    const cudaDataType_t bias_type = is_float_bias ? CUDA_R_32F : type;
    OMNI_RETURN_VAL_IF_LOG(
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                       sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                           &bias_type, sizeof(bias_type)) != CUBLAS_STATUS_SUCCESS,
        false, ERROR, "Failed to configure the BLAS bias epilogue");
  }
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
                                   const std::array<cudaEvent_t, 2U> &events,
                                   const bool enable_tuning, void *const output) {
  OMNI_RETURN_VAL_IF(handle == nullptr || input == nullptr || weight == nullptr ||
                         workspace.data == nullptr || events[0] == nullptr ||
                         events[1] == nullptr || output == nullptr,
                     false);
  std::array<cublasLtMatmulHeuristicResult_t, 8U> candidates = {};
  int32_t result_count = 0;
  OMNI_RETURN_VAL_IF_LOG(
      cublasLtMatmulAlgoGetHeuristic(handle, operation, layouts[0], layouts[1], layouts[2],
                                     layouts[2], preference, enable_tuning ? candidates.size() : 1U,
                                     candidates.data(), &result_count) != CUBLAS_STATUS_SUCCESS ||
          result_count == 0,
      false, ERROR, "No FP32-accumulating BLAS linear algorithm");
  if (!enable_tuning) {
    OMNI_RETURN_VAL_IF_LOG(candidates[0].state != CUBLAS_STATUS_SUCCESS, false, ERROR,
                           "Invalid BLAS linear heuristic candidate");
    algorithm = candidates[0].algo;
    return true;
  }
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
  if (parallel_runtime_ == nullptr) {
    auto runtime = std::make_unique<ParallelRuntime>();
    OMNI_RETURN_VAL_IF(!runtime->Initialize(), false);
    parallel_runtime_ = std::move(runtime);
  }
  return true;
}

size_t CudaBackend::compute_workspace_bytes() const {
  size_t bytes = 0U;
  for (const auto &[key, workspace] : quantized_workspaces_) {
    bytes += workspace->buffers[0].size_bytes + workspace->buffers[1].size_bytes;
  }
  for (const auto &[key, workspace] : fp8_workspaces_) {
    bytes += workspace->buffers[0].size_bytes + workspace->buffers[1].size_bytes;
  }
  for (const auto &[key, workspace] : biased_workspaces_) {
    bytes += workspace->buffers[0].size_bytes;
  }
  for (const auto &[key, workspace] : delta_workspaces_) {
    bytes += workspace->buffers[0].size_bytes;
  }
  return linear_workspace_.size_bytes + bytes +
         (parallel_runtime_ == nullptr ? 0U : 2U * ParallelRuntime::kWorkspaceBytes);
}

bool CudaBackend::Fp8Matmul(const ScalarType type, const Fp8Linear &operation) {
  OMNI_RETURN_VAL_IF_LOG(!kernels::runtime::ValidateFp8(operation, ElementBytes(type)), false,
                         ERROR, "Invalid FP8 E4M3FN block-linear buffers or dimensions");
  const std::lock_guard<std::mutex> lock(compute_mutex_);
  const std::array<uint64_t, 3U> key = {
      reinterpret_cast<uintptr_t>(operation.output.data), operation.output.offset_bytes,
      (static_cast<uint64_t>(operation.rows) << 32U) | static_cast<uint32_t>(operation.inner)};
  auto iterator = fp8_workspaces_.find(key);
  if (iterator == fp8_workspaces_.end()) {
    OMNI_RETURN_VAL_IF_LOG(kernel::jit::JitRuntime::stream() != nullptr &&
                               !kernel::jit::JitRuntime::is_preparing(),
                           false, ERROR, "FP8 workspace must be prepared before capture");
    const size_t values = static_cast<size_t>(operation.rows) * operation.inner;
    const size_t activation_bytes = values * (operation.rows <= 8 ? sizeof(int32_t) : sizeof(uint8_t));
    auto workspace = std::make_unique<MatmulWorkspace>();
    OMNI_RETURN_VAL_IF_LOG(
        !Allocate(MemoryKind::kDevice, activation_bytes, &workspace->buffers[0]) ||
            !Allocate(MemoryKind::kDevice, values / 128U * sizeof(float), &workspace->buffers[1]),
        false, ERROR, "Failed to allocate FP8 activation workspace");
    iterator = fp8_workspaces_.emplace(key, std::move(workspace)).first;
  }
  const auto &buffers = iterator->second->buffers;
  return DispatchScalarType(type, [&]<typename T>() {
    return kernels::runtime::CudaOperator<T, Fp8Linear>::Launch(
        operation, static_cast<uint8_t *>(buffers[0].data), static_cast<float *>(buffers[1].data));
  });
}

bool CudaBackend::ChunkDeltaNet(const GatedDeltaNet &operation) {
  OMNI_RETURN_VAL_IF_LOG(!kernels::runtime::ValidateDelta(operation, sizeof(BFloat16)) ||
                             operation.head_size != 128 ||
                             operation.token_count > std::numeric_limits<int32_t>::max() - 63,
                         false,
                         ERROR, "Invalid chunked delta net inputs");
  const std::lock_guard<std::mutex> lock(compute_mutex_);
  const int32_t heads = operation.value_head_count;
  const int32_t size = operation.head_size;
  const int32_t chunks = (operation.token_count + 63) / 64;
  OMNI_RETURN_VAL_IF_LOG(heads > std::numeric_limits<int32_t>::max() / chunks, false, ERROR,
                         "Chunked delta net batch count exceeds int32");
  const int32_t padded = chunks * 64;
  const size_t values = static_cast<size_t>(heads) * padded * size;
  const size_t matrices = static_cast<size_t>(heads) * padded * 64;
  const size_t state_values = static_cast<size_t>(heads) * size * size;
  const size_t state_offset = 7 * values + 2 * matrices + static_cast<size_t>(heads) * padded;
  const std::array<uint64_t, 3U> key = {
      reinterpret_cast<uintptr_t>(operation.output.data), operation.output.offset_bytes,
      (static_cast<uint64_t>(heads) << 32U) | static_cast<uint32_t>(padded)};
  auto iterator = delta_workspaces_.find(key);
  if (iterator == delta_workspaces_.end()) {
    OMNI_RETURN_VAL_IF_LOG(kernel::jit::JitRuntime::stream() != nullptr &&
                               !kernel::jit::JitRuntime::is_preparing(),
                           false, ERROR, "Delta net workspace must be prepared before capture");
    auto storage = std::make_unique<MatmulWorkspace>();
    OMNI_RETURN_VAL_IF(!Allocate(MemoryKind::kDevice,
                                   (state_offset + 2 * state_values) * sizeof(float),
                                   &storage->buffers[0]), false);
    iterator = delta_workspaces_.emplace(key, std::move(storage)).first;
  }
  float *const workspace = static_cast<float *>(iterator->second->buffers[0].data);
  float *const query = workspace;
  float *const key_values = workspace + values;
  float *const key_beta = workspace + 2 * values;
  float *const new_values = workspace + 4 * values;
  float *const inter = workspace + 5 * values;
  float *const intra = workspace + 6 * values;
  float *const system = workspace + 7 * values;
  float *const attention = system + matrices;
  float *const state = workspace + state_offset;
  float *const next_state = state + state_values;
  const auto launch = [&]<typename... Args>(const std::string &name, const dim3 grid,
                                           const uint32_t threads, const Args &...args) {
    return kernels::runtime::CudaKernelRuntime::CompileAndLaunch(
        name, kernels::runtime::CudaKernelRuntime::MakeKernelSource<__nv_bfloat16>(
                  "kernels/delta_net_kernel/delta_net_kernel.cuh",
                  "omni_runtime::kernels::delta_net_kernel::" + name),
        {.grid_dim = grid, .block_dim = {threads, 1U, 1U}, .shared_memory_bytes = 0}, args...);
  };
  OMNI_RETURN_VAL_IF(!launch("PrepareChunkImpl", {static_cast<uint32_t>(heads),
                                                   static_cast<uint32_t>(padded), 1U}, 32U,
                             CudaAddress<const __nv_bfloat16>(operation.qkv),
                             CudaAddress<const __nv_bfloat16>(operation.alpha),
                             CudaAddress<const __nv_bfloat16>(operation.beta),
                             CudaAddress<const float>(operation.decay_weight),
                             CudaAddress<const float>(operation.time_bias), operation.token_count,
                             operation.key_head_count, heads, size, operation.epsilon,
                             operation.is_log_decay, workspace),
                     false);
  const cudaStream_t stream = NativeStream(kernel::jit::JitRuntime::stream());
  const bool is_preparing = kernel::jit::JitRuntime::is_preparing();
  const float one = 1.0F;
  const float zero = 0.0F;
  const auto gemm = [&](const cublasOperation_t first_op, const cublasOperation_t second_op,
                         const int32_t rows, const int32_t columns, const int32_t inner,
                         const float scale, const float *const first, const int32_t first_leading,
                         const size_t first_stride, const float *const second,
                         const int32_t second_leading, const size_t second_stride,
                         const float accumulation, const int32_t output_leading,
                         const size_t output_stride, const int32_t batches, float *const output) {
    OMNI_RETURN_VAL_IF(is_preparing, true);
    const cublasStatus_t status = batches == 1
        ? cublasSgemm(blas_handle_, first_op, second_op, rows, columns, inner, &scale,
                       first, first_leading, second, second_leading, &accumulation,
                       output, output_leading)
        : cublasSgemmStridedBatched(
                               blas_handle_, first_op, second_op, rows, columns, inner, &scale,
                               first, first_leading, first_stride, second, second_leading,
                               second_stride, &accumulation, output, output_leading, output_stride,
                               batches);
    OMNI_RETURN_VAL_IF_LOG(status != CUBLAS_STATUS_SUCCESS,
                           false, ERROR, "Chunked delta net matrix multiplication failed");
    return true;
  };
  const auto execute = [&] {
    OMNI_RETURN_VAL_IF_LOG(!is_preparing &&
                               (cublasSetStream(blas_handle_, stream) != CUBLAS_STATUS_SUCCESS ||
                                cublasSetWorkspace(blas_handle_, linear_workspace_.data,
                                                   linear_workspace_.size_bytes) !=
                                    CUBLAS_STATUS_SUCCESS),
                           false, ERROR, "Failed to configure chunked delta net BLAS");
    OMNI_RETURN_VAL_IF(
        !gemm(CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, size, one, key_values, size, 64 * size,
              key_beta, size, 64 * size, zero, 64, 4096, heads * chunks, system) ||
            !gemm(CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, size, one, key_values, size, 64 * size,
                  query, size, 64 * size, zero, 64, 4096, heads * chunks, attention) ||
            !launch("TransformChunkImpl", {static_cast<uint32_t>(heads * chunks), 1U, 1U},
                    256U, padded, heads, size, workspace) ||
            !launch("SolveChunkImpl", {static_cast<uint32_t>(heads * chunks), 2U, 1U},
                    128U, padded, heads, size, workspace),
        false);
    const dim3 state_grid{static_cast<uint32_t>((state_values + 255U) / 256U), 1U, 1U};
    OMNI_RETURN_VAL_IF(!launch("ChunkStateImpl", state_grid, 256U,
                                CudaAddress<const float>(operation.state), heads, size, state),
                       false);
    for (int32_t chunk = 0; chunk < chunks; ++chunk) {
      const size_t offset = static_cast<size_t>(chunk) * 64 * size;
      OMNI_RETURN_VAL_IF(
          !gemm(CUBLAS_OP_N, CUBLAS_OP_N, size, 64, size, one, state, size, size * size,
                key_beta + offset, size, padded * size, zero, size, padded * size, heads,
                new_values + offset) ||
              !launch("SubtractChunkImpl", {static_cast<uint32_t>(heads), 1U, 1U}, 256U,
                      padded, heads, size, chunk, workspace) ||
              !gemm(CUBLAS_OP_N, CUBLAS_OP_N, size, 64, size, one, state, size, size * size,
                    query + offset, size, padded * size, zero, size, padded * size, heads,
                    inter + offset) ||
              !gemm(CUBLAS_OP_N, CUBLAS_OP_N, size, 64, 64, one, new_values + offset, size,
                    padded * size, attention + chunk * 4096, 64, chunks * 4096, zero,
                    size, padded * size, heads, intra + offset) ||
              !gemm(CUBLAS_OP_N, CUBLAS_OP_T, size, size, 64, one, new_values + offset, size,
                    padded * size, key_values + offset, size, padded * size, zero,
                    size, size * size, heads, next_state) ||
              !launch("FinishChunkImpl", {static_cast<uint32_t>(heads), 1U, 1U}, 256U,
                      operation.token_count, heads, size, chunk, workspace,
                      CudaAddress<__nv_bfloat16>(operation.output)),
          false);
    }
    return launch("ChunkStateImpl", state_grid, 256U, state, heads, size,
                  CudaAddress<float>(operation.state));
  };
  const bool is_successful = execute();
  OMNI_RETURN_VAL_IF_LOG(!is_preparing &&
                             cublasSetStream(blas_handle_, nullptr) != CUBLAS_STATUS_SUCCESS,
                         false, ERROR, "Failed to restore the default delta net BLAS stream");
  return is_successful;
}

bool CudaBackend::QuantizedMatmul(const QuantizedLinear &operation) {
  OMNI_RETURN_VAL_IF_LOG(!kernels::runtime::ValidateQuantized(operation, sizeof(float)), false,
                         ERROR, "Invalid mixed-storage linear multiplication");
  const std::lock_guard<std::mutex> lock(compute_mutex_);
  const uint64_t input_count = static_cast<uint64_t>(operation.rows) * operation.inner;
  const uint64_t weight_count =
      operation.weight_type == 40 ? static_cast<uint64_t>(operation.columns) * operation.inner : 0U;
  const std::array<size_t, 2U> required = {input_count * sizeof(uint16_t),
                                           weight_count * sizeof(uint16_t)};
  const std::array<uint64_t, 2U> key = {input_count, weight_count};
  auto iterator = quantized_workspaces_.find(key);
  if (iterator == quantized_workspaces_.end()) {
    OMNI_RETURN_VAL_IF_LOG(
        kernel::jit::JitRuntime::stream() != nullptr && !kernel::jit::JitRuntime::is_preparing(),
        false, ERROR, "Quantized BLAS workspaces must be prepared before capture");
    auto workspace = std::make_unique<MatmulWorkspace>();
    for (size_t index = 0U; index < required.size(); ++index) {
      OMNI_RETURN_VAL_IF(required[index] > 0U && !Allocate(MemoryKind::kDevice, required[index],
                                                           &workspace->buffers[index]),
                         false);
    }
    iterator = quantized_workspaces_.emplace(key, std::move(workspace)).first;
  }
  const auto &buffers = iterator->second->buffers;
  const uint64_t count = std::max(input_count, weight_count);
  OMNI_RETURN_VAL_IF(!kernels::runtime::CudaKernelRuntime::CompileAndLaunch(
                         "gguf_prepare_bfloat",
                         kernels::runtime::CudaKernelRuntime::MakeKernelSource<float>(
                             "kernels/quantized_kernel/quantized_kernel.cuh",
                             "omni_runtime::kernels::quantized_kernel::PrepareBfloatImpl"),
                         {.grid_dim = {static_cast<uint32_t>((count + 255U) / 256U), 1U, 1U},
                          .block_dim = {256U, 1U, 1U},
                          .shared_memory_bytes = 0},
                         CudaAddress<const float>(operation.input),
                         CudaAddress<const uint8_t>(operation.weight), input_count, weight_count,
                         buffers[0].data, buffers[1].data),
                     false);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  void *const output = CudaAddress<void>(operation.output);
  const void *const weight =
      weight_count == 0U ? CudaAddress<void>(operation.weight) : buffers[1].data;
  // NVFP4's E4M3 scales times E2M1 codes are exactly representable in BF16.
  if (!kernel::jit::JitRuntime::is_preparing()) {
    OMNI_RETURN_VAL_IF_LOG(
        cublasSetStream(blas_handle_, NativeStream(kernel::jit::JitRuntime::stream())) !=
                CUBLAS_STATUS_SUCCESS ||
            cublasSetWorkspace(blas_handle_, linear_workspace_.data,
                               linear_workspace_.size_bytes) != CUBLAS_STATUS_SUCCESS,
        false, ERROR, "Failed to configure mixed-storage BLAS");
    const cublasStatus_t status = cublasGemmEx(
        blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, operation.columns, operation.rows, operation.inner,
        &alpha, weight, CUDA_R_16BF, operation.inner, buffers[0].data, CUDA_R_16BF, operation.inner,
        &beta, output, CUDA_R_32F, operation.columns, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    const cublasStatus_t reset_status = cublasSetStream(blas_handle_, nullptr);
    OMNI_RETURN_VAL_IF_LOG(
        status != CUBLAS_STATUS_SUCCESS, false, ERROR,
        "Mixed-storage BLAS multiplication failed: " << static_cast<int>(status));
    OMNI_RETURN_VAL_IF_LOG(reset_status != CUBLAS_STATUS_SUCCESS, false, ERROR,
                           "Failed to restore the default BLAS stream");
  }
  if (!operation.activation_scales.IsValid() && operation.weight_scale == 1.0F) {
    return true;
  }
  const uint64_t output_count = static_cast<uint64_t>(operation.rows) * operation.columns;
  return kernels::runtime::CudaKernelRuntime::CompileAndLaunch(
      "gguf_scale_rows",
      kernels::runtime::CudaKernelRuntime::MakeKernelSource<float>(
          "kernels/quantized_kernel/quantized_kernel.cuh",
          "omni_runtime::kernels::quantized_kernel::ScaleRowsImpl"),
      {.grid_dim = {static_cast<uint32_t>((output_count + 255U) / 256U), 1U, 1U},
       .block_dim = {256U, 1U, 1U},
       .shared_memory_bytes = 0},
      operation.activation_scales.IsValid() ? CudaAddress<const float>(operation.activation_scales)
                                            : nullptr,
      operation.activation_scales.IsValid() ? operation.inner / operation.activation_block_size : 0,
      operation.weight_scale, operation.columns, output_count, output);
}

bool CudaBackend::LinearMatmul(const ScalarType type, const DeviceView &input_view,
                               const DeviceView &weight_view, const int32_t rows,
                               const int32_t inner, const int32_t columns,
                               const DeviceView &output_view) {
  OMNI_RETURN_VAL_IF_LOG(!input_view.IsValid() || !weight_view.IsValid() || !output_view.IsValid(),
                         false, ERROR, "Invalid BLAS linear multiplication views");
  const void *const input = CudaAddress<void>(input_view);
  const void *const weight = CudaAddress<void>(weight_view);
  void *const output = CudaAddress<void>(output_view);
  OMNI_RETURN_VAL_IF_LOG(input == nullptr || weight == nullptr || output == nullptr ||
                             blas_handle_ == nullptr || rows <= 0 || inner <= 0 || columns <= 0 ||
                             (type != ScalarType::kFloat32 && type != ScalarType::kFloat16 &&
                              type != ScalarType::kBFloat16),
                         false, ERROR, "Invalid BLAS linear multiplication inputs");
  if (rows == 1 && type != ScalarType::kBFloat16) {
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
  const cudaStream_t stream = NativeStream(kernel::jit::JitRuntime::stream());
  if (stream != nullptr) {
    OMNI_RETURN_VAL_IF_LOG(type != ScalarType::kFloat32 && type != ScalarType::kBFloat16, false,
                           ERROR,
                           "Captured multirow linear operations require FP32 or BF16 storage");
    if (kernel::jit::JitRuntime::is_preparing()) {
      if (type == ScalarType::kBFloat16) {
        return LaunchLinear(type,
                            {.input = input_view,
                             .weight = weight_view,
                             .rows = rows,
                             .inner = inner,
                             .columns = columns,
                             .output = output_view},
                            blas_handle_, linear_workspace_, nullptr);
      }
      return true;
    }
    OMNI_RETURN_VAL_IF_LOG(cublasSetStream(blas_handle_, stream) != CUBLAS_STATUS_SUCCESS ||
                               cublasSetWorkspace(blas_handle_, linear_workspace_.data,
                                                  linear_workspace_.size_bytes) !=
                                   CUBLAS_STATUS_SUCCESS,
                           false, ERROR, "Failed to configure captured FP32 BLAS");
    const bool is_launched = LaunchLinear(type,
                                          {.input = input_view,
                                           .weight = weight_view,
                                           .rows = rows,
                                           .inner = inner,
                                           .columns = columns,
                                           .output = output_view},
                                          blas_handle_, linear_workspace_, stream);
    const cublasStatus_t reset_status = cublasSetStream(blas_handle_, nullptr);
    OMNI_RETURN_VAL_IF_LOG(reset_status != CUBLAS_STATUS_SUCCESS, false, ERROR,
                           "Failed to restore the default FP32 BLAS stream");
    return is_launched;
  }
  OMNI_RETURN_VAL_IF_LOG(type == ScalarType::kBFloat16 &&
                             cublasSetWorkspace(blas_handle_, linear_workspace_.data,
                                                linear_workspace_.size_bytes) !=
                                 CUBLAS_STATUS_SUCCESS,
                         false, ERROR, "Failed to configure BF16 BLAS workspace");
  return LaunchLinear(type,
                      {.input = input_view,
                       .weight = weight_view,
                       .rows = rows,
                       .inner = inner,
                       .columns = columns,
                       .output = output_view},
                      blas_handle_, linear_workspace_, nullptr);
}

bool CudaBackend::BiasedLinear(const ScalarType type, const Linear &operation) {
  const size_t bytes = ElementBytes(type);
  const size_t bias_bytes = operation.is_float_bias ? sizeof(float) : bytes;
  OMNI_RETURN_VAL_IF_LOG(
      !HasLinearPlan(type, operation) ||
          (!operation.is_float_bias && type != ScalarType::kBFloat16 &&
           type != ScalarType::kFloat32) ||
          !HasElements(operation.input, uint64_t(operation.rows) * operation.inner, bytes) ||
          !HasElements(operation.weight, uint64_t(operation.columns) * operation.inner, bytes) ||
          !HasElements(operation.output, uint64_t(operation.rows) * operation.columns, bytes) ||
          !HasElements(operation.bias, operation.columns, bias_bytes) ||
          CudaAddress<const void>(operation.bias) == nullptr,
      false, ERROR, "Invalid biased linear buffers or dimensions");
  const cudaStream_t stream = NativeStream(kernel::jit::JitRuntime::stream());
  const std::lock_guard<std::mutex> lock(compute_mutex_);
  if (type == ScalarType::kBFloat16 && !operation.is_float_bias) {
    return LaunchLinear(type, operation, blas_handle_, linear_workspace_,
                        kernel::jit::JitRuntime::is_preparing() ? nullptr : stream);
  }
  const uint64_t count = static_cast<uint64_t>(operation.rows) * operation.columns;
  const size_t output_bytes = count * sizeof(float);
  const std::array<uint64_t, 2U> key = {reinterpret_cast<uintptr_t>(operation.output.data),
                                        operation.output.offset_bytes};
  auto iterator = biased_workspaces_.find(key);
  if (iterator == biased_workspaces_.end() ||
      iterator->second->buffers[0].size_bytes < output_bytes) {
    OMNI_RETURN_VAL_IF_LOG(stream != nullptr && !kernel::jit::JitRuntime::is_preparing(), false,
                           ERROR, "Biased linear workspace must be prepared before capture");
    auto workspace = std::make_unique<MatmulWorkspace>();
    OMNI_RETURN_VAL_IF(!Allocate(MemoryKind::kDevice, output_bytes, &workspace->buffers[0]), false);
    iterator = biased_workspaces_.insert_or_assign(key, std::move(workspace)).first;
  }
  const Allocation &output = iterator->second->buffers[0];
  Linear projection = operation;
  projection.output = {
      .data = output.data, .size_bytes = output.size_bytes, .backend_kind = BackendKind::kCuda};
  OMNI_RETURN_VAL_IF(!LaunchLinear(type, projection, blas_handle_, linear_workspace_,
                                   kernel::jit::JitRuntime::is_preparing() ? nullptr : stream),
                     false);
  return DispatchScalarType(type, [&]<typename T>() {
    return kernels::runtime::CudaKernelRuntime::CompileAndLaunch(
        "linear_convert_output",
        kernels::runtime::CudaKernelRuntime::MakeKernelSource<T>(
            "kernels/matmul_kernel/matmul_kernel.cuh",
            "omni_runtime::kernels::matmul_kernel::LinearConvertOutputImpl"),
        {.grid_dim = {static_cast<uint32_t>((count + 255U) / 256U), 1U, 1U},
         .block_dim = {256U, 1U, 1U},
         .shared_memory_bytes = 0},
        static_cast<const float *>(output.data), count, CudaAddress<T>(operation.output));
  });
}

bool CudaBackend::LaunchLinear(const ScalarType type, const Linear &operation,
                               const cublasHandle_t handle, const Allocation &workspace,
                               const cudaStream_t stream) {
  const bool has_bias = operation.bias.IsValid();
  const void *const input = CudaAddress<void>(has_bias ? operation.weight : operation.input);
  const void *const weight = CudaAddress<void>(has_bias ? operation.input : operation.weight);
  void *const output = CudaAddress<void>(operation.output);
  const int32_t rows = operation.rows;
  const int32_t inner = operation.inner;
  const int32_t columns = operation.columns;
  const cudaDataType_t data_type = type == ScalarType::kFloat32   ? CUDA_R_32F
                                   : type == ScalarType::kFloat16 ? CUDA_R_16F
                                                                  : CUDA_R_16BF;
  const float alpha = 1.0F;
  const float beta = 0.0F;
  if (HasLinearPlan(type, operation)) {
    const std::array<int32_t, 6U> key = {static_cast<int32_t>(type), rows, inner, columns,
                                         static_cast<int32_t>(has_bias),
                                         static_cast<int32_t>(operation.is_float_bias)};
    auto iterator = linear_plans_.find(key);
    if (iterator == linear_plans_.end()) {
      OMNI_RETURN_VAL_IF_LOG(stream != nullptr, false, ERROR,
                             "Parallel BLAS algorithms must be prepared before submission");
      auto plan = std::make_unique<LinearPlan>();
      OMNI_RETURN_VAL_IF(!plan->Initialize(linear_handle_, data_type, rows, inner, columns,
                                           linear_workspace_.size_bytes, has_bias,
                                           operation.is_float_bias),
                         false);
      if (has_bias) {
        const void *const bias = CudaAddress<void>(operation.bias);
        OMNI_RETURN_VAL_IF_LOG(
            cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                           &bias, sizeof(bias)) != CUBLAS_STATUS_SUCCESS,
            false, ERROR, "Failed to bind the BLAS linear bias");
      }
      OMNI_RETURN_VAL_IF(
          !plan->Tune(linear_handle_, input, weight, linear_workspace_, linear_events_,
                       type != ScalarType::kBFloat16 || operation.is_float_bias, output),
          false);
      iterator = linear_plans_.emplace(key, std::move(plan)).first;
    }
    const LinearPlan &plan = *iterator->second;
    if (has_bias) {
      const void *const bias = CudaAddress<void>(operation.bias);
      OMNI_RETURN_VAL_IF_LOG(
          cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                         sizeof(bias)) != CUBLAS_STATUS_SUCCESS,
          false, ERROR, "Failed to update the BLAS linear bias");
    }
    OMNI_RETURN_VAL_IF_LOG(cublasLtMatmul(linear_handle_, plan.operation, &alpha, input,
                                          plan.layouts[0], weight, plan.layouts[1], &beta, output,
                                          plan.layouts[2], output, plan.layouts[2], &plan.algorithm,
                                          workspace.data, workspace.size_bytes,
                                          stream) != CUBLAS_STATUS_SUCCESS,
                           false, ERROR, "BLAS linear plan execution failed");
    return true;
  }
  const cublasStatus_t status =
      kernel::jit::JitRuntime::is_preparing() ? CUBLAS_STATUS_SUCCESS :
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, inner, &alpha, weight,
                   data_type, inner, input, data_type, inner, &beta, output, data_type, columns,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  OMNI_RETURN_VAL_IF_LOG(status != CUBLAS_STATUS_SUCCESS, false, ERROR,
                         "BLAS linear multiplication failed: " << static_cast<int>(status));
  return true;
}

bool CudaBackend::ParallelMatmul(const ScalarType type, const ParallelLinear &operation) {
  OMNI_RETURN_VAL_IF_LOG(parallel_runtime_ == nullptr, false, ERROR,
                         "Parallel CUDA execution is not initialized");
  const std::lock_guard<std::mutex> lock(compute_mutex_);
  for (const Linear &projection : operation.projections) {
    const std::array<int32_t, 6U> key = {static_cast<int32_t>(type), projection.rows,
                                         projection.inner, projection.columns,
                                         static_cast<int32_t>(projection.bias.IsValid()),
                                         static_cast<int32_t>(projection.is_float_bias)};
    if (HasLinearPlan(type, projection) && linear_plans_.find(key) == linear_plans_.end()) {
      OMNI_RETURN_VAL_IF(!LaunchLinear(type, projection, blas_handle_, linear_workspace_, nullptr),
                         false);
    }
  }
  ParallelRuntime &runtime = *parallel_runtime_;
  const auto submit = [&](const bool is_parallel) {
    if (!is_parallel) {
      for (const Linear &projection : operation.projections) {
        OMNI_RETURN_VAL_IF(
            !LaunchLinear(type, projection, blas_handle_, linear_workspace_, nullptr), false);
      }
      return true;
    }
    OMNI_RETURN_VAL_IF_LOG(cudaEventRecord(runtime.ready, nullptr) != cudaSuccess, false, ERROR,
                           "Failed to fork CUDA projections");
    for (size_t branch = 1U; branch < operation.projections.size(); ++branch) {
      const size_t lane = branch - 1U;
      OMNI_RETURN_VAL_IF_LOG(
          cudaStreamWaitEvent(runtime.streams[lane], runtime.ready, 0U) != cudaSuccess ||
              !LaunchLinear(type, operation.projections[branch], runtime.handles[lane],
                            runtime.workspaces[lane], runtime.streams[lane]) ||
              cudaEventRecord(runtime.completed[lane], runtime.streams[lane]) != cudaSuccess,
          false, ERROR, "Parallel CUDA projection submission failed");
    }
    OMNI_RETURN_VAL_IF(
        !LaunchLinear(type, operation.projections[0], blas_handle_, linear_workspace_, nullptr),
        false);
    for (size_t branch = 1U; branch < operation.projections.size(); ++branch) {
      OMNI_RETURN_VAL_IF_LOG(cudaStreamWaitEvent(nullptr, runtime.completed[branch - 1U], 0U) !=
                                 cudaSuccess,
                             false, ERROR, "Failed to join CUDA projections");
    }
    return true;
  };
  const auto execute = [&] {
    const Linear &first = operation.projections.front();
    const std::array<int32_t, 6U> key = {
        static_cast<int32_t>(type),
        first.rows,
        first.inner,
        first.columns,
        operation.projections[1].columns,
        operation.projections.size() == 3U ? operation.projections[2].columns : 0};
    auto strategy = parallel_strategies_.find(key);
    // Large batches already saturate the GPU; avoid six serial/parallel timing rounds at prefill.
    if (strategy == parallel_strategies_.end() && first.rows >= kParallelProjectionTuneMaxRows) {
      strategy = parallel_strategies_.emplace(key, false).first;
      LOG_INFO("CUDA projection strategy rows=" << first.rows << ", inner=" << first.inner
                                                << ", branches=" << operation.projections.size()
                                                << ": serial (large-batch policy)");
    }
    if (strategy == parallel_strategies_.end()) {
      OMNI_RETURN_VAL_IF(!submit(false) || !submit(true), false);
      std::array<float, 3U> serial_times_ms = {};
      std::array<float, 3U> parallel_times_ms = {};
      for (size_t iteration = 0U; iteration < serial_times_ms.size(); ++iteration) {
        for (const bool is_parallel : {iteration % 2U != 0U, iteration % 2U == 0U}) {
          OMNI_RETURN_VAL_IF_LOG(cudaEventRecord(linear_events_[0], nullptr) != cudaSuccess ||
                                     !submit(is_parallel) ||
                                     cudaEventRecord(linear_events_[1], nullptr) != cudaSuccess ||
                                     cudaEventSynchronize(linear_events_[1]) != cudaSuccess,
                                 false, ERROR, "Failed to measure parallel projection strategy");
          float *const elapsed_ms =
              is_parallel ? &parallel_times_ms[iteration] : &serial_times_ms[iteration];
          OMNI_RETURN_VAL_IF_LOG(
              cudaEventElapsedTime(elapsed_ms, linear_events_[0], linear_events_[1]) != cudaSuccess,
              false, ERROR, "Failed to time parallel projection strategy");
        }
      }
      std::sort(serial_times_ms.begin(), serial_times_ms.end());
      std::sort(parallel_times_ms.begin(), parallel_times_ms.end());
      const bool is_parallel = parallel_times_ms[1] * 1.05F < serial_times_ms[1];
      strategy = parallel_strategies_.emplace(key, is_parallel).first;
      LOG_INFO("CUDA projection strategy rows=" << first.rows << ", inner=" << first.inner
                                                << ", branches=" << operation.projections.size()
                                                << ": " << (is_parallel ? "parallel" : "serial")
                                                << ", serial_ms=" << serial_times_ms[1]
                                                << ", parallel_ms=" << parallel_times_ms[1]);
    }
    return submit(strategy->second);
  };
  if (!execute()) {
    const bool is_drained = runtime.Drain();
    const cudaError_t status = cudaStreamSynchronize(nullptr);
    if (!is_drained || status != cudaSuccess) {
      LOG_ERROR("Failed to drain partially submitted CUDA projections");
    }
    return false;
  }
  return true;
}

} // namespace backend
} // namespace omni_runtime
