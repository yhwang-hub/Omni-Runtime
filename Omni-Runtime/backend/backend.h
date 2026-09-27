#pragma once

#include <cstddef>
#include <memory>

#include "Omni-Runtime/backend/backend_define.h"

namespace omni_runtime {
namespace backend {

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
  virtual bool DestroyStream(void *stream) = 0;
};

std::shared_ptr<Backend> CreateBackend(const BackendKind backend_kind);
bool IsBackendAvailable(const BackendKind backend_kind);

} // namespace backend
} // namespace omni_runtime
