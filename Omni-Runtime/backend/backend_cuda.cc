#include "Omni-Runtime/backend/backend_cuda.h"

#include <cuda_runtime.h>

#include "Omni-Runtime/backend/backend_cuda_utils.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace backend {

using cuda_internal::AllocationAddress;
using cuda_internal::NativeEvent;
using cuda_internal::NativeStream;

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
