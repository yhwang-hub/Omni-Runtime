#pragma once

#include <memory>

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/backend/backend_define.h"

namespace omni_runtime {
namespace memory {

class BackendEvent;

class BackendStream final {
public:
  static std::shared_ptr<BackendStream> Create(const std::shared_ptr<backend::Backend> &backend,
                                               const uint32_t flags);
  ~BackendStream();
  BackendStream(const BackendStream &) = delete;
  BackendStream &operator=(const BackendStream &) = delete;

  bool Synchronize() const;
  bool Query(bool *const is_ready) const;
  bool WaitEvent(const BackendEvent &event, const uint32_t flags) const;
  const void *native_handle() const;
  backend::BackendKind backend_kind() const;
  const std::shared_ptr<backend::Backend> &backend() const;

private:
  explicit BackendStream(const std::shared_ptr<backend::Backend> &backend);
  std::shared_ptr<backend::Backend> backend_;
  backend::StreamHandle stream_handle_;
};

using BackendStreamPtr = std::shared_ptr<BackendStream>;

} // namespace memory
} // namespace omni_runtime
