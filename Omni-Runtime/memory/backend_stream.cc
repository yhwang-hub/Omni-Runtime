#include "Omni-Runtime/memory/backend_stream.h"

#include "Omni-Runtime/memory/backend_event.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace memory {

BackendStream::BackendStream(const std::shared_ptr<backend::Backend> &backend) : backend_(backend) {
}

BackendStream::~BackendStream() {
  if (backend_ != nullptr && stream_handle_.native_handle != nullptr) {
    static_cast<void>(backend_->DestroyStream(stream_handle_.native_handle));
  }
}

std::shared_ptr<BackendStream>
BackendStream::Create(const std::shared_ptr<backend::Backend> &backend, const uint32_t flags) {
  OMNI_RETURN_VAL_IF(backend == nullptr, nullptr);
  auto stream = std::shared_ptr<BackendStream>(new BackendStream(backend));
  OMNI_RETURN_VAL_IF(!backend->CreateStream(flags, &stream->stream_handle_), nullptr);
  return stream;
}

bool BackendStream::Synchronize() const {
  return backend_ != nullptr && backend_->SynchronizeStream(stream_handle_.native_handle);
}

bool BackendStream::Query(bool *const is_ready) const {
  OMNI_RETURN_VAL_IF(is_ready == nullptr, false);
  return backend_ != nullptr && backend_->QueryStream(stream_handle_.native_handle, is_ready);
}

bool BackendStream::WaitEvent(const BackendEvent &event, const uint32_t flags) const {
  OMNI_RETURN_VAL_IF(backend_ == nullptr || event.backend() != backend_, false);
  return backend_->WaitEvent(stream_handle_.native_handle, event.event_handle().native_handle,
                             flags);
}

const void *BackendStream::native_handle() const {
  return stream_handle_.native_handle;
}

backend::BackendKind BackendStream::backend_kind() const {
  return stream_handle_.backend_kind;
}

const std::shared_ptr<backend::Backend> &BackendStream::backend() const {
  return backend_;
}

} // namespace memory
} // namespace omni_runtime
