#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "Omni-Runtime/memory/buffer.h"

namespace omni_runtime::memory {

enum class DataType : uint8_t { kFloat32 = 0, kFloat16 = 1, kBFloat16 = 2, kInt32 = 3, kUInt8 = 4 };
std::size_t SizeOf(DataType type);

class Tensor final {
public:
  static std::shared_ptr<Tensor> Create(const std::shared_ptr<backend::Backend> &backend,
                                        const std::vector<std::size_t> &shape, DataType type,
                                        backend::MemoryKind memory_kind);
  const std::vector<std::size_t> &shape() const;
  DataType type() const;
  std::size_t element_count() const;
  std::size_t size_bytes() const;
  const BufferPtr &buffer() const;
  bool Reshape(const std::vector<std::size_t> &shape);
  bool CopyFromHost(const void *const source, std::size_t size_bytes);
  bool CopyToHost(void *const destination, std::size_t size_bytes) const;

private:
  Tensor(const BufferPtr &buffer, std::vector<std::size_t> shape, DataType type);
  BufferPtr buffer_;
  std::vector<std::size_t> shape_;
  DataType type_ = DataType::kFloat32;
};
using TensorPtr = std::shared_ptr<Tensor>;

} // namespace omni_runtime::memory
