#include "Omni-Runtime/memory/tensor.h"

#include <numeric>

namespace omni_runtime {
namespace memory {
std::size_t SizeOf(const DataType type) {
  switch (type) {
  case DataType::kFloat32:
  case DataType::kInt32:
    return 4U;
  case DataType::kFloat16:
  case DataType::kBFloat16:
    return 2U;
  case DataType::kUInt8:
    return 1U;
  }
  return 0U;
}
Tensor::Tensor(const BufferPtr &buffer, std::vector<std::size_t> shape, const DataType type)
    : buffer_(buffer), shape_(std::move(shape)), type_(type) {
}
std::shared_ptr<Tensor> Tensor::Create(const std::shared_ptr<backend::Backend> &backend,
                                       const std::vector<std::size_t> &shape, const DataType type,
                                       const backend::MemoryKind memory_kind) {
  const std::size_t element_count =
      std::accumulate(shape.begin(), shape.end(), std::size_t{1U}, std::multiplies<std::size_t>());
  const std::size_t bytes = element_count * SizeOf(type);
  if (shape.empty() || bytes == 0U) {
    return nullptr;
  }
  const BufferPtr buffer = Buffer::CreateOwnedBuffer(backend, bytes, memory_kind);
  return buffer == nullptr ? nullptr : std::shared_ptr<Tensor>(new Tensor(buffer, shape, type));
}
const std::vector<std::size_t> &Tensor::shape() const {
  return shape_;
}
DataType Tensor::type() const {
  return type_;
}
std::size_t Tensor::element_count() const {
  return size_bytes() / SizeOf(type_);
}
std::size_t Tensor::size_bytes() const {
  return buffer_ == nullptr ? 0U : buffer_->size_bytes();
}
const BufferPtr &Tensor::buffer() const {
  return buffer_;
}
bool Tensor::Reshape(const std::vector<std::size_t> &shape) {
  if (shape.empty()) {
    return false;
  }
  const std::size_t count =
      std::accumulate(shape.begin(), shape.end(), std::size_t{1U}, std::multiplies<std::size_t>());
  if (count != element_count()) {
    return false;
  }
  shape_ = shape;
  return true;
}
bool Tensor::CopyFromHost(const void *const source, const std::size_t bytes) {
  return buffer_ != nullptr && bytes == this->size_bytes() && buffer_->CopyFromHost(source, bytes);
}
bool Tensor::CopyToHost(void *const destination, const std::size_t bytes) const {
  return buffer_ != nullptr && bytes == this->size_bytes() &&
         buffer_->CopyToHost(bytes, destination);
}
} // namespace memory
} // namespace omni_runtime
