#pragma once

#include <cstdint>
#include <string>

namespace omni_runtime {
namespace inference {

enum class ModelTask : uint8_t {
  kTextGeneration,
  kActionPrediction,
};

struct ModelDescriptor final {
  std::string architecture;
  ModelTask task = ModelTask::kTextGeneration;
  bool supports_images = false;
};

} // namespace inference
} // namespace omni_runtime
