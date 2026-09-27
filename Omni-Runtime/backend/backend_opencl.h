#pragma once

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <functional>
#include <memory>

#include "Omni-Runtime/backend/backend.h"

namespace omni_runtime {
namespace backend {

class OpenCLBackend final : public Backend {
public:
  OpenCLBackend();
  ~OpenCLBackend() override;

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
  bool DestroyStream(void *stream) override;

private:
  bool Submit(const void *stream, const std::function<cl_int(cl_command_queue)> &operation);

  struct Runtime;
  std::unique_ptr<Runtime> runtime_;
};

} // namespace backend
} // namespace omni_runtime
