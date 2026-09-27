#include "Omni-Runtime/backend/backend.h"

#include "Omni-Runtime/backend/backend_cuda.h"
#include "Omni-Runtime/backend/backend_opencl.h"
#include "Omni-Runtime/utils/logger.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace backend {

EventTimer::~EventTimer() {
  if (backend_ != nullptr) {
    const bool is_start_destroyed = backend_->DestroyEvent(start_.native_handle);
    const bool is_end_destroyed = backend_->DestroyEvent(end_.native_handle);
    if (!is_start_destroyed || !is_end_destroyed) {
      LOG_ERROR("Failed to destroy backend timing events");
    }
  }
}

bool EventTimer::Initialize(const std::shared_ptr<Backend> &backend) {
  OMNI_RETURN_VAL_IF_LOG(backend == nullptr || backend_ != nullptr, false, ERROR,
                         "Invalid backend timer initialization");
  backend_ = backend;
  return backend_->CreateEvent(kDefaultEventFlags, &start_) &&
         backend_->CreateEvent(kDefaultEventFlags, &end_);
}

bool EventTimer::Start() {
  OMNI_RETURN_VAL_IF_LOG(backend_ == nullptr, false, ERROR, "Backend timer is not initialized");
  is_started_ = backend_->RecordEvent(nullptr, &start_);
  return is_started_;
}

bool EventTimer::Stop(double *const elapsed_ms) {
  OMNI_RETURN_VAL_IF_LOG(elapsed_ms == nullptr || !is_started_, false, ERROR,
                         "Backend timer has not started or output is null");
  is_started_ = false;
  return backend_->RecordEvent(nullptr, &end_) && backend_->SynchronizeEvent(end_.native_handle) &&
         backend_->ElapsedTimeMs(start_.native_handle, end_.native_handle, elapsed_ms);
}

bool Backend::InitializeCompute() {
  LOG_ERROR("Inference compute is unsupported by this backend");
  return false;
}

bool Backend::Compute(const ScalarType type, const ComputeOperation &operation) {
  LOG_ERROR("Requested compute operation is unsupported by this backend");
  return false;
}

size_t Backend::compute_workspace_bytes() const {
  return 0U;
}

bool Backend::LinearMatmul(const ScalarType type, const void *const input, const void *const weight,
                           const int32_t rows, const int32_t inner, const int32_t columns,
                           void *const output) {
  LOG_ERROR("Linear matrix multiplication is unsupported by this backend");
  return false;
}

std::shared_ptr<Backend> CreateBackend(const BackendKind backend_kind) {
  if (backend_kind == BackendKind::kAuto) {
    for (const BackendKind candidate : {BackendKind::kCuda, BackendKind::kOpenCL}) {
      const std::shared_ptr<Backend> backend = CreateBackend(candidate);
      if (backend != nullptr && backend->IsAvailable()) {
        return backend;
      }
    }
    LOG_ERROR("No available compute backend");
    return nullptr;
  }
  if (backend_kind == BackendKind::kCuda) {
    return std::make_shared<CudaBackend>();
  }
  if (backend_kind == BackendKind::kOpenCL) {
    return std::make_shared<OpenCLBackend>();
  }
  return nullptr;
}

bool IsBackendAvailable(const BackendKind backend_kind) {
  const std::shared_ptr<Backend> backend = CreateBackend(backend_kind);
  return backend != nullptr && backend->IsAvailable();
}

const char *BackendName(const BackendKind backend_kind) {
  switch (backend_kind) {
  case BackendKind::kCuda:
    return "cuda";
  case BackendKind::kOpenCL:
    return "opencl";
  case BackendKind::kAuto:
    return "auto";
  }
  return "unknown";
}

} // namespace backend
} // namespace omni_runtime
