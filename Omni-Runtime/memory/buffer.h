#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/backend/backend_define.h"
#include "Omni-Runtime/memory/backend_stream.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace memory {

class Buffer final {
public:
  static std::shared_ptr<Buffer> CreateOwnedBuffer(const std::shared_ptr<backend::Backend> &backend,
                                                   const std::size_t size_bytes,
                                                   const backend::MemoryKind memory_kind);
  static std::shared_ptr<Buffer> CreateHostBufferView(void *const source,
                                                      const std::size_t size_bytes);
  static std::shared_ptr<Buffer>
  CreateDeviceBufferView(const std::shared_ptr<backend::Backend> &backend, void *const source,
                         const std::size_t size_bytes);

  template <typename T>
  static std::shared_ptr<Buffer>
  CreateDeviceBuffer(const std::shared_ptr<backend::Backend> &backend,
                     const std::size_t element_count) {
    OMNI_RETURN_VAL_IF(backend == nullptr || element_count == 0U, nullptr);
    OMNI_RETURN_VAL_IF(element_count > std::numeric_limits<std::size_t>::max() / sizeof(T),
                       nullptr);
    return CreateOwnedBuffer(backend, element_count * sizeof(T), backend::MemoryKind::kDevice);
  }

  template <typename T>
  static std::shared_ptr<Buffer>
  CreateDeviceBufferFromVector(const std::shared_ptr<backend::Backend> &backend,
                               const std::vector<T> &source) {
    const auto buffer = CreateDeviceBuffer<T>(backend, source.size());
    OMNI_RETURN_VAL_IF(buffer == nullptr, nullptr);
    OMNI_RETURN_VAL_IF(!buffer->CopyDataFromVector(source), nullptr);
    return buffer;
  }

  ~Buffer();
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  bool CopyFromHost(const void *const source, const std::size_t size_bytes);
  bool CopyFromHostAsync(const void *const source, const std::size_t size_bytes,
                         const BackendStream &stream);
  bool CopyToHost(const std::size_t size_bytes, void *const destination) const;
  bool CopyToHostAsync(const std::size_t size_bytes, void *const destination,
                       const BackendStream &stream) const;
  bool CopyFromBuffer(const Buffer &source);
  bool CopyFromBufferAsync(const Buffer &source, const BackendStream &stream);
  bool Fill(const uint8_t value);
  bool FillAsync(const uint8_t value, const BackendStream &stream);

  template <typename T> bool CopyDataFromVector(const std::vector<T> &source) {
    return source.size() <= std::numeric_limits<std::size_t>::max() / sizeof(T) &&
           CopyFromHost(source.data(), source.size() * sizeof(T));
  }

  template <typename T> bool CopyDataToVector(std::vector<T> *const destination) const {
    OMNI_RETURN_VAL_IF(destination == nullptr, false);
    destination->resize(size_bytes() / sizeof(T));
    return CopyToHost(size_bytes(), destination->data());
  }

  std::size_t size_bytes() const;
  backend::MemoryKind memory_kind() const;
  bool is_host_accessible() const;
  void *host_address() const;
  void *device_address() const;

  template <typename T> std::size_t element_count() const {
    const std::size_t bytes = size_bytes();
    return bytes % sizeof(T) == 0U ? bytes / sizeof(T) : 0U;
  }

  const backend::Allocation &allocation() const;
  const std::shared_ptr<backend::Backend> &backend() const;

private:
  // stream may be nullptr for synchronous execution.
  bool CopyFromHostOnStream(const void *const source, const std::size_t size_bytes,
                            const void *const stream);
  bool CopyToHostOnStream(const std::size_t size_bytes, void *const destination,
                          const void *const stream) const;
  bool CopyFromBufferOnStream(const Buffer &source, const void *const stream);
  bool FillOnStream(const uint8_t value, const void *const stream);
  void *mutable_address() const;

  Buffer(const std::shared_ptr<backend::Backend> &backend, const std::size_t size_bytes,
         const backend::MemoryKind memory_kind);
  Buffer(void *external_data, const std::size_t size_bytes, const backend::MemoryKind memory_kind);

  std::shared_ptr<backend::Backend> backend_;
  backend::Allocation allocation_;
  std::vector<uint8_t> host_storage_;
  void *external_data_ = nullptr;
  backend::MemoryKind memory_kind_ = backend::MemoryKind::kHost;
  bool is_owned_ = false;
};

using BufferPtr = std::shared_ptr<Buffer>;

} // namespace memory
} // namespace omni_runtime
