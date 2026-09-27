#include "Omni-Runtime/memory/backend_event.h"

#include "Omni-Runtime/memory/backend_stream.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace memory {

BackendEvent::BackendEvent(const std::shared_ptr<backend::Backend> &backend) : backend_(backend) {
}

BackendEvent::~BackendEvent() {
  if (backend_ != nullptr && event_handle_.native_handle != nullptr) {
    static_cast<void>(backend_->DestroyEvent(event_handle_.native_handle));
  }
}

std::shared_ptr<BackendEvent> BackendEvent::Create(const std::shared_ptr<backend::Backend> &backend,
                                                   const uint32_t flags) {
  OMNI_RETURN_VAL_IF(backend == nullptr, nullptr);
  auto event = std::shared_ptr<BackendEvent>(new BackendEvent(backend));
  OMNI_RETURN_VAL_IF(!backend->CreateEvent(flags, &event->event_handle_), nullptr);
  return event;
}

bool BackendEvent::Record(const BackendStream &stream) {
  OMNI_RETURN_VAL_IF(backend_ == nullptr || stream.backend() != backend_, false);
  return backend_->RecordEvent(stream.native_handle(), &event_handle_);
}

bool BackendEvent::Synchronize() const {
  return backend_ != nullptr && backend_->SynchronizeEvent(event_handle_.native_handle);
}

bool BackendEvent::Query(bool *const is_ready) const {
  OMNI_RETURN_VAL_IF(is_ready == nullptr, false);
  return backend_ != nullptr && backend_->QueryEvent(event_handle_.native_handle, is_ready);
}

const backend::EventHandle &BackendEvent::event_handle() const {
  return event_handle_;
}

const std::shared_ptr<backend::Backend> &BackendEvent::backend() const {
  return backend_;
}

} // namespace memory
} // namespace omni_runtime
