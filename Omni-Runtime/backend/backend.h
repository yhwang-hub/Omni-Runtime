#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "Omni-Runtime/backend/backend_define.h"
#include "Omni-Runtime/backend/backend_compute_define.h"

namespace omni_runtime {
namespace backend {

bool CompileComputeGraph(const ScalarType type, const BackendKind kind,
                         const std::vector<ComputeNode> &nodes,
                         std::vector<ComputeOperation> *const operations);
bool GetParallelExecutionEnabled(bool *const is_enabled);

class ExecutionPlan {
public:
  ExecutionPlan() = default;
  virtual ~ExecutionPlan() = default;
  ExecutionPlan(const ExecutionPlan &) = delete;
  ExecutionPlan &operator=(const ExecutionPlan &) = delete;

  // Bound allocations must outlive the plan; callers serialize Run invocations.
  // The device result is copied before Run returns; elapsed_ms excludes readback.
  virtual bool Run(const size_t result_bytes, void *const result, double *const elapsed_ms) = 0;
};

class Backend {
public:
  Backend() = default;
  virtual ~Backend() = default;

  Backend(const Backend &) = delete;
  Backend &operator=(const Backend &) = delete;

  virtual BackendKind kind() const = 0;
  virtual bool IsAvailable() const = 0;
  virtual bool IsMemoryKindSupported(const MemoryKind memory_kind) const = 0;
  virtual bool Allocate(const MemoryKind memory_kind, const std::size_t size_bytes,
                        Allocation *const allocation) = 0;
  virtual bool Free(Allocation *const allocation) = 0;
  virtual bool Fill(const Allocation &allocation, const uint8_t value, const void *stream) = 0;
  virtual bool CopyHostToDevice(const void *const source, const std::size_t size_bytes,
                                const Allocation &destination, const void *stream) = 0;
  virtual bool CopyDeviceToHost(const Allocation &source, const std::size_t size_bytes,
                                void *const destination, const void *stream) = 0;
  virtual bool CopyDeviceToDevice(const Allocation &source, const Allocation &destination,
                                  const std::size_t size_bytes, const void *stream) = 0;
  virtual bool CreateStream(const uint32_t flags, StreamHandle *const stream_handle) = 0;
  virtual bool SynchronizeStream(const void *stream) = 0;
  virtual bool QueryStream(const void *stream, bool *const is_ready) = 0;
  virtual bool WaitEvent(const void *stream, const void *event, const uint32_t flags) = 0;
  virtual bool CreateEvent(const uint32_t flags, EventHandle *const event_handle) = 0;
  virtual bool DestroyEvent(void *event) = 0;
  virtual bool RecordEvent(const void *stream, EventHandle *const event_handle) = 0;
  virtual bool SynchronizeEvent(const void *event) = 0;
  virtual bool QueryEvent(const void *event, bool *const is_ready) = 0;
  virtual bool ElapsedTimeMs(const void *const start, const void *const end,
                             double *const elapsed_ms) = 0;
  virtual bool DestroyStream(void *stream) = 0;
  virtual bool InitializeCompute();
  virtual bool Compute(const ScalarType type, const ComputeOperation &operation);
  virtual bool CreateExecutionPlan(const ScalarType type,
                                   const std::vector<ComputeOperation> &operations,
                                   const Allocation &readback,
                                   std::unique_ptr<ExecutionPlan> *const plan);
  bool CreateExecutionPlan(const ScalarType type, const std::vector<ComputeNode> &nodes,
                           const Allocation &readback, std::unique_ptr<ExecutionPlan> *const plan);
  virtual size_t compute_workspace_bytes() const;
  virtual bool LinearMatmul(const ScalarType type, const DeviceView &input,
                            const DeviceView &weight, const int32_t rows, const int32_t inner,
                            const int32_t columns, const DeviceView &output);
};

class EventTimer final {
public:
  EventTimer() = default;
  ~EventTimer();
  EventTimer(const EventTimer &) = delete;
  EventTimer &operator=(const EventTimer &) = delete;

  bool Initialize(const std::shared_ptr<Backend> &backend);
  bool Start();
  bool Stop(double *const elapsed_ms);

private:
  std::shared_ptr<Backend> backend_;
  EventHandle start_;
  EventHandle end_;
  bool is_started_ = false;
};

std::shared_ptr<Backend> CreateBackend(const BackendKind backend_kind);
std::shared_ptr<Backend> CreateComputeBackend(const std::string &name);
bool IsBackendAvailable(const BackendKind backend_kind);
const char *BackendName(const BackendKind backend_kind);

} // namespace backend
} // namespace omni_runtime
