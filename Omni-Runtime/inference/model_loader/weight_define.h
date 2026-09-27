#pragma once

#include <cstdint>
#include <vector>

#include "Omni-Runtime/gguf_serializer/GGUF.h"
#include "Omni-Runtime/memory/buffer.h"

namespace omni_runtime {
namespace inference {

struct WeightStorage final {
  memory::BufferPtr storage;
  std::vector<uint64_t> dimensions;
  util::GGUFTensorType type = util::GGUFTensorType::FLOAT32;
  float scale = 1.0F;
};

} // namespace inference
} // namespace omni_runtime
