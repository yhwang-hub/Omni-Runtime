#pragma once

#include <functional>
#include <array>
#include <map>
#include <memory>
#include <mutex>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "Omni-Runtime/backend/backend.h"

namespace omni_runtime {
namespace backend {

class CudaBackend final : public Backend {
public:
  CudaBackend();
  CudaBackend(const CudaBackend &) = delete;
  CudaBackend &operator=(const CudaBackend &) = delete;
  ~CudaBackend() override;

  using Backend::CreateExecutionPlan;

  BackendKind kind() const override;
  bool IsAvailable() const override;
  bool IsMemoryKindSupported(const MemoryKind memory_kind) const override;
  bool Allocate(const MemoryKind memory_kind, const std::size_t size_bytes,
                Allocation *const allocation) override;
  bool Free(Allocation *const allocation) override;
  bool Fill(const Allocation &allocation, const uint8_t value, const void *stream) override;
  bool CopyHostToDevice(const void *const source, const std::size_t size_bytes,
                        const Allocation &destination, const void *stream) override;
  bool CopyDeviceToHost(const Allocation &source, const std::size_t size_bytes,
                        void *const destination, const void *stream) override;
  bool CopyDeviceToDevice(const Allocation &source, const Allocation &destination,
                          const std::size_t size_bytes, const void *stream) override;
  bool CreateStream(const uint32_t flags, StreamHandle *const stream_handle) override;
  bool SynchronizeStream(const void *stream) override;
  bool QueryStream(const void *stream, bool *const is_ready) override;
  bool WaitEvent(const void *stream, const void *event, const uint32_t flags) override;
  bool CreateEvent(const uint32_t flags, EventHandle *const event_handle) override;
  bool DestroyEvent(void *event) override;
  bool RecordEvent(const void *stream, EventHandle *const event_handle) override;
  bool SynchronizeEvent(const void *event) override;
  bool QueryEvent(const void *event, bool *const is_ready) override;
  bool ElapsedTimeMs(const void *const start, const void *const end,
                     double *const elapsed_ms) override;
  bool DestroyStream(void *stream) override;
  bool InitializeCompute() override;
  bool Compute(const ScalarType type, const ComputeOperation &operation) override;
  bool CreateExecutionPlan(const ScalarType type, const std::vector<ComputeOperation> &operations,
                           const Allocation &readback,
                           std::unique_ptr<ExecutionPlan> *const plan) override;
  size_t compute_workspace_bytes() const override;
  bool LinearMatmul(const ScalarType type, const DeviceView &input_view,
                    const DeviceView &weight_view, const int32_t rows, const int32_t inner,
                    const int32_t columns, const DeviceView &output_view) override;

private:
  class CudaExecutionPlan;
  struct ParallelRuntime;

  struct LinearPlan final {
    ~LinearPlan();
    bool Initialize(const cublasLtHandle_t handle, const cudaDataType_t type, const int32_t rows,
                    const int32_t inner, const int32_t columns, const size_t workspace_bytes);
    bool Tune(const cublasLtHandle_t handle, const void *const input, const void *const weight,
              const Allocation &workspace, const std::array<cudaEvent_t, 2U> &events,
              void *const output);

    cublasLtMatmulDesc_t operation = nullptr;
    std::array<cublasLtMatrixLayout_t, 3U> layouts = {};
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulAlgo_t algorithm = {};
  };

  bool Submit(const void *stream, const std::function<cudaError_t(cudaStream_t)> &operation);
  bool ParallelMatmul(const ScalarType type, const ParallelLinear &operation);
  bool LaunchLinear(const ScalarType type, const Linear &operation, const cublasHandle_t handle,
                    const Allocation &workspace, const cudaStream_t stream);
  cublasHandle_t blas_handle_ = nullptr;
  std::mutex compute_mutex_;
  cublasLtHandle_t linear_handle_ = nullptr;
  Allocation linear_workspace_;
  std::array<cudaEvent_t, 2U> linear_events_ = {};
  std::map<std::array<int32_t, 4U>, std::unique_ptr<LinearPlan>> linear_plans_;
  std::map<std::array<int32_t, 6U>, bool> parallel_strategies_;
  std::unique_ptr<ParallelRuntime> parallel_runtime_;
  bool is_jit_initialized_ = false;
};

} // namespace backend
} // namespace omni_runtime
