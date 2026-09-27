#include "Omni-Runtime/memory/buffer.h"

#include <algorithm>
#include <cstring>

namespace omni_runtime {
namespace memory {
namespace {

bool IsHostAccessible(const backend::MemoryKind memory_kind) {
  return memory_kind == backend::MemoryKind::kHost ||
         memory_kind == backend::MemoryKind::kHostView ||
         memory_kind == backend::MemoryKind::kPinnedHost ||
         memory_kind == backend::MemoryKind::kManaged;
}

bool IsDeviceMemory(const backend::MemoryKind memory_kind) {
  return memory_kind == backend::MemoryKind::kDevice ||
         memory_kind == backend::MemoryKind::kDeviceView;
}

} // namespace

Buffer::Buffer(const std::shared_ptr<backend::Backend> &backend, const std::size_t size_bytes,
               const backend::MemoryKind memory_kind)
    : backend_(backend), memory_kind_(memory_kind), is_owned_(true) {
  if (memory_kind == backend::MemoryKind::kHost) {
    host_storage_.resize(size_bytes);
  }
}

Buffer::Buffer(void *external_data, const std::size_t size_bytes,
               const backend::MemoryKind memory_kind)
    : external_data_(external_data), memory_kind_(memory_kind) {
}

Buffer::~Buffer() {
  if (!is_owned_ || memory_kind_ == backend::MemoryKind::kHost || backend_ == nullptr) {
    return;
  }
  static_cast<void>(backend_->Free(&allocation_));
}

std::shared_ptr<Buffer> Buffer::CreateOwnedBuffer(const std::shared_ptr<backend::Backend> &backend,
                                                  const std::size_t size_bytes,
                                                  const backend::MemoryKind memory_kind) {
  OMNI_RETURN_VAL_IF(size_bytes == 0U, nullptr);
  if (memory_kind == backend::MemoryKind::kHost) {
    auto buffer = std::shared_ptr<Buffer>(
        new Buffer(std::shared_ptr<backend::Backend>(), size_bytes, memory_kind));
    buffer->allocation_.size_bytes = size_bytes;
    return buffer;
  }
  OMNI_RETURN_VAL_IF(backend == nullptr || !backend->IsMemoryKindSupported(memory_kind), nullptr);

  auto buffer = std::shared_ptr<Buffer>(new Buffer(backend, size_bytes, memory_kind));
  OMNI_RETURN_VAL_IF(!backend->Allocate(memory_kind, size_bytes, &buffer->allocation_), nullptr);
  return buffer;
}

std::shared_ptr<Buffer> Buffer::CreateHostBufferView(void *const source,
                                                     const std::size_t size_bytes) {
  OMNI_RETURN_VAL_IF(source == nullptr || size_bytes == 0U, nullptr);
  auto buffer =
      std::shared_ptr<Buffer>(new Buffer(source, size_bytes, backend::MemoryKind::kHostView));
  buffer->allocation_.size_bytes = size_bytes;
  return buffer;
}

std::shared_ptr<Buffer>
Buffer::CreateDeviceBufferView(const std::shared_ptr<backend::Backend> &backend, void *const source,
                               const std::size_t size_bytes) {
  OMNI_RETURN_VAL_IF(backend == nullptr || source == nullptr || size_bytes == 0U, nullptr);
  auto buffer =
      std::shared_ptr<Buffer>(new Buffer(nullptr, size_bytes, backend::MemoryKind::kDeviceView));
  buffer->backend_ = backend;
  buffer->allocation_.data = source;
  buffer->allocation_.size_bytes = size_bytes;
  buffer->allocation_.memory_kind = backend::MemoryKind::kDeviceView;
  return buffer;
}

bool Buffer::CopyFromHost(const void *const source, const std::size_t size_bytes) {
  return CopyFromHostOnStream(source, size_bytes, nullptr);
}

bool Buffer::CopyFromHostAsync(const void *const source, const std::size_t size_bytes,
                               const BackendStream &stream) {
  return CopyFromHostOnStream(source, size_bytes, stream.native_handle());
}

bool Buffer::CopyToHost(const std::size_t size_bytes, void *const destination) const {
  return CopyToHostOnStream(size_bytes, destination, nullptr);
}

bool Buffer::CopyToHostAsync(const std::size_t size_bytes, void *const destination,
                             const BackendStream &stream) const {
  return CopyToHostOnStream(size_bytes, destination, stream.native_handle());
}

bool Buffer::CopyFromBuffer(const Buffer &source) {
  return CopyFromBufferOnStream(source, nullptr);
}

bool Buffer::CopyFromBufferAsync(const Buffer &source, const BackendStream &stream) {
  return CopyFromBufferOnStream(source, stream.native_handle());
}

bool Buffer::Fill(const uint8_t value) {
  return FillOnStream(value, nullptr);
}

bool Buffer::FillAsync(const uint8_t value, const BackendStream &stream) {
  return FillOnStream(value, stream.native_handle());
}

bool Buffer::CopyFromHostOnStream(const void *const source, const std::size_t size_bytes,
                                  const void *const stream) {
  OMNI_RETURN_VAL_IF(source == nullptr || size_bytes > this->size_bytes(), false);
  if (is_host_accessible()) {
    std::memcpy(mutable_address(), source, size_bytes);
    return true;
  }
  return backend_ != nullptr && backend_->CopyHostToDevice(source, size_bytes, allocation_, stream);
}

bool Buffer::CopyToHostOnStream(const std::size_t size_bytes, void *const destination,
                                const void *const stream) const {
  OMNI_RETURN_VAL_IF(destination == nullptr || size_bytes > this->size_bytes(), false);
  if (is_host_accessible()) {
    std::memcpy(destination, mutable_address(), size_bytes);
    return true;
  }
  return backend_ != nullptr &&
         backend_->CopyDeviceToHost(allocation_, size_bytes, destination, stream);
}

bool Buffer::CopyFromBufferOnStream(const Buffer &source, const void *const stream) {
  OMNI_RETURN_VAL_IF(source.size_bytes() != size_bytes(), false);
  const std::size_t bytes = size_bytes();
  if (source.is_host_accessible() && is_host_accessible()) {
    std::memcpy(mutable_address(), source.mutable_address(), bytes);
    return true;
  }
  if (source.is_host_accessible()) {
    return backend_ != nullptr &&
           backend_->CopyHostToDevice(source.mutable_address(), bytes, allocation_, stream);
  }
  if (is_host_accessible()) {
    return source.backend_ != nullptr &&
           source.backend_->CopyDeviceToHost(source.allocation_, bytes, mutable_address(), stream);
  }
  return backend_ != nullptr && source.backend_ != nullptr &&
         backend_->CopyDeviceToDevice(source.allocation_, allocation_, bytes, stream);
}

bool Buffer::FillOnStream(const uint8_t value, const void *const stream) {
  if (is_host_accessible()) {
    std::fill(reinterpret_cast<uint8_t *>(mutable_address()),
              reinterpret_cast<uint8_t *>(mutable_address()) + size_bytes(), value);
    return true;
  }
  return backend_ != nullptr && backend_->Fill(allocation_, value, stream);
}

std::size_t Buffer::size_bytes() const {
  if (memory_kind_ == backend::MemoryKind::kHost) {
    return host_storage_.size();
  }
  if (memory_kind_ == backend::MemoryKind::kHostView) {
    return external_data_ == nullptr ? 0U : allocation_.size_bytes;
  }
  return allocation_.size_bytes;
}

backend::MemoryKind Buffer::memory_kind() const {
  return memory_kind_;
}

bool Buffer::is_host_accessible() const {
  return IsHostAccessible(memory_kind_);
}

void *Buffer::host_address() const {
  return is_host_accessible() ? mutable_address() : nullptr;
}

void *Buffer::device_address() const {
  return IsDeviceMemory(memory_kind_) ? allocation_.data : nullptr;
}

void *Buffer::mutable_address() const {
  if (memory_kind_ == backend::MemoryKind::kHost) {
    return const_cast<uint8_t *>(host_storage_.data());
  }
  if (memory_kind_ == backend::MemoryKind::kHostView) {
    return external_data_;
  }
  return allocation_.data;
}

const backend::Allocation &Buffer::allocation() const {
  return allocation_;
}

const std::shared_ptr<backend::Backend> &Buffer::backend() const {
  return backend_;
}

} // namespace memory
} // namespace omni_runtime
