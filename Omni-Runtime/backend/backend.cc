#include "Omni-Runtime/backend/backend.h"

#include "Omni-Runtime/backend/backend_cuda.h"
#include "Omni-Runtime/backend/backend_opencl.h"

namespace omni_runtime {
namespace backend {

std::shared_ptr<Backend> CreateBackend(const BackendKind backend_kind) {
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

} // namespace backend
} // namespace omni_runtime
