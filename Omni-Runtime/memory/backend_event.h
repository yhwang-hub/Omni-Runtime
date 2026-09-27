#pragma once

#include <memory>

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/backend/backend_define.h"

namespace omni_runtime {
namespace memory {

class BackendStream;

class BackendEvent final {
public:
  static std::shared_ptr<BackendEvent> Create(const std::shared_ptr<backend::Backend> &backend,
                                              const uint32_t flags);
  ~BackendEvent();
  BackendEvent(const BackendEvent &) = delete;
  BackendEvent &operator=(const BackendEvent &) = delete;

  bool Record(const BackendStream &stream);
  bool Synchronize() const;
  bool Query(bool *const is_ready) const;
  const backend::EventHandle &event_handle() const;
  const std::shared_ptr<backend::Backend> &backend() const;

private:
  explicit BackendEvent(const std::shared_ptr<backend::Backend> &backend);
  std::shared_ptr<backend::Backend> backend_;
  backend::EventHandle event_handle_;
};

using BackendEventPtr = std::shared_ptr<BackendEvent>;

} // namespace memory
} // namespace omni_runtime
