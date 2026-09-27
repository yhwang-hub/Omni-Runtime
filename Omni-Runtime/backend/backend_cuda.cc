#include "Omni-Runtime/backend/backend_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <cstring>
#include "Omni-Runtime/backend/runtime_path.h"
#include "Omni-Runtime/kernels/compute_runtime.h"

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

void *AllocationAddress(const Allocation &allocation) {
  return static_cast<uint8_t *>(allocation.data) + allocation.offset_bytes;
}

bool HasLinearPlan(const ScalarType type, const Linear &operation) {
  return type != ScalarType::kFloat32 && operation.rows > 1 && operation.inner % 8 == 0 &&
         operation.columns % 8 == 0 &&
         reinterpret_cast<uintptr_t>(kernels::runtime::CudaAddress<void>(operation.input)) % 16U ==
             0U &&
         reinterpret_cast<uintptr_t>(kernels::runtime::CudaAddress<void>(operation.weight)) % 16U ==
             0U &&
         reinterpret_cast<uintptr_t>(kernels::runtime::CudaAddress<void>(operation.output)) % 16U ==
             0U;
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

bool CudaBackend::Compute(const ScalarType type, const ComputeOperation &operation) {
  OMNI_RETURN_VAL_IF_LOG(!is_jit_initialized_, false, ERROR, "Compute backend is not initialized");
  const bool is_successful = DispatchScalarType(type, [&]<typename T>() {
    return std::visit(
        [this, type](const auto &args) {
          if constexpr (std::is_same_v<std::decay_t<decltype(args)>, Linear>) {
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
          } else {
            return kernels::runtime::OperatorRuntime<T, std::decay_t<decltype(args)>>::Launch(
                BackendKind::kCuda, args, nullptr);
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
    OMNI_RETURN_VAL_IF_LOG(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
                                   cudaSuccess ||
                               cudaMallocHost(&result_, readback.size_bytes) != cudaSuccess ||
                               cudaEventCreate(&events_[0]) != cudaSuccess ||
                               cudaEventCreate(&events_[1]) != cudaSuccess,
                           false, ERROR, "Failed to allocate execution plan resources");
    const auto launch = [&] {
      for (const ComputeOperation &operation : operations) {
        OMNI_RETURN_VAL_IF(!backend->Compute(type, operation), false);
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
  bool Execute(void *const result, double *const elapsed_ms) {
    OMNI_RETURN_VAL_IF_LOG(cudaEventRecord(events_[0], stream_) != cudaSuccess ||
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
    OMNI_RETURN_VAL_IF_LOG(linear != nullptr && linear->rows != 1, false, ERROR,
                           "Prepared linear operations require one row");
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
  if (parallel_runtime_ == nullptr) {
    auto runtime = std::make_unique<ParallelRuntime>();
    OMNI_RETURN_VAL_IF(!runtime->Initialize(), false);
    parallel_runtime_ = std::move(runtime);
  }
  return true;
}

size_t CudaBackend::compute_workspace_bytes() const {
  return linear_workspace_.size_bytes +
         (parallel_runtime_ == nullptr ? 0U : 2U * ParallelRuntime::kWorkspaceBytes);
}

bool CudaBackend::LinearMatmul(const ScalarType type, const DeviceView &input_view,
                               const DeviceView &weight_view, const int32_t rows,
                               const int32_t inner, const int32_t columns,
                               const DeviceView &output_view) {
  OMNI_RETURN_VAL_IF_LOG(!input_view.IsValid() || !weight_view.IsValid() || !output_view.IsValid(),
                         false, ERROR, "Invalid BLAS linear multiplication views");
  const void *const input = kernels::runtime::CudaAddress<void>(input_view);
  const void *const weight = kernels::runtime::CudaAddress<void>(weight_view);
  void *const output = kernels::runtime::CudaAddress<void>(output_view);
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
  return LaunchLinear(type,
                      {.input = input_view,
                       .weight = weight_view,
                       .rows = rows,
                       .inner = inner,
                       .columns = columns,
                       .output = output_view},
                      blas_handle_, linear_workspace_, nullptr);
}

bool CudaBackend::LaunchLinear(const ScalarType type, const Linear &operation,
                               const cublasHandle_t handle, const Allocation &workspace,
                               const cudaStream_t stream) {
  const void *const input = kernels::runtime::CudaAddress<void>(operation.input);
  const void *const weight = kernels::runtime::CudaAddress<void>(operation.weight);
  void *const output = kernels::runtime::CudaAddress<void>(operation.output);
  const int32_t rows = operation.rows;
  const int32_t inner = operation.inner;
  const int32_t columns = operation.columns;
  const cudaDataType_t data_type = type == ScalarType::kFloat32   ? CUDA_R_32F
                                   : type == ScalarType::kFloat16 ? CUDA_R_16F
                                                                  : CUDA_R_16BF;
  const float alpha = 1.0F;
  const float beta = 0.0F;
  if (HasLinearPlan(type, operation)) {
    const std::array<int32_t, 4U> key = {static_cast<int32_t>(type), rows, inner, columns};
    auto iterator = linear_plans_.find(key);
    if (iterator == linear_plans_.end()) {
      OMNI_RETURN_VAL_IF_LOG(stream != nullptr, false, ERROR,
                             "Parallel BLAS algorithms must be prepared before submission");
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
                                          workspace.data, workspace.size_bytes,
                                          stream) != CUBLAS_STATUS_SUCCESS,
                           false, ERROR, "BLAS linear plan execution failed");
    return true;
  }
  const cublasStatus_t status =
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
    const std::array<int32_t, 4U> key = {static_cast<int32_t>(type), projection.rows,
                                         projection.inner, projection.columns};
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
  allocation->offset_bytes = 0U;
  return true;
}

bool CudaBackend::Free(Allocation *const allocation) {
  OMNI_RETURN_VAL_IF(allocation == nullptr, false);
  OMNI_RETURN_VAL_IF(allocation->offset_bytes != 0U, false);
  OMNI_RETURN_VAL_IF(allocation->data == nullptr, true);
  const cudaError_t status = cudaFree(allocation->data);
  allocation->data = nullptr;
  allocation->size_bytes = 0U;
  allocation->offset_bytes = 0U;
  return status == cudaSuccess;
}

bool CudaBackend::Fill(const Allocation &allocation, const uint8_t value, const void *stream) {
  OMNI_RETURN_VAL_IF(allocation.data == nullptr, false);
  return Submit(stream, [&](cudaStream_t native_stream) {
    return cudaMemsetAsync(AllocationAddress(allocation), value, allocation.size_bytes,
                           native_stream);
  });
}

bool CudaBackend::CopyHostToDevice(const void *const source, const std::size_t size_bytes,
                                   const Allocation &destination, const void *stream) {
  OMNI_RETURN_VAL_IF(source == nullptr || destination.data == nullptr ||
                         size_bytes > destination.size_bytes,
                     false);
  return Submit(stream, [&](cudaStream_t native_stream) {
    return cudaMemcpyAsync(AllocationAddress(destination), source, size_bytes,
                           cudaMemcpyHostToDevice, native_stream);
  });
}

bool CudaBackend::CopyDeviceToHost(const Allocation &source, const std::size_t size_bytes,
                                   void *const destination, const void *stream) {
  OMNI_RETURN_VAL_IF(
      destination == nullptr || source.data == nullptr || size_bytes > source.size_bytes, false);
  return Submit(stream, [&](cudaStream_t native_stream) {
    return cudaMemcpyAsync(destination, AllocationAddress(source), size_bytes,
                           cudaMemcpyDeviceToHost, native_stream);
  });
}

bool CudaBackend::CopyDeviceToDevice(const Allocation &source, const Allocation &destination,
                                     const std::size_t size_bytes, const void *stream) {
  OMNI_RETURN_VAL_IF(source.data == nullptr || destination.data == nullptr ||
                         size_bytes > source.size_bytes || size_bytes > destination.size_bytes,
                     false);
  return Submit(stream, [&](cudaStream_t native_stream) {
    return cudaMemcpyAsync(AllocationAddress(destination), AllocationAddress(source), size_bytes,
                           cudaMemcpyDeviceToDevice, native_stream);
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
