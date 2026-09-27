#pragma once

#include <unordered_map>

#include "Omni-Runtime/inference/common/inference_config/inference_config.h"
#include "Omni-Runtime/memory/tensor.h"
#include "Omni-Runtime/models/qwen3_vl/config_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

struct Qwen3VLModelData final {
  std::shared_ptr<backend::Backend> backend;
  InferenceConfig config;
  Qwen3VLConfig model_config;
  qwen3::Qwen3VisionConfig vision_config;
  std::unordered_map<std::string, memory::TensorPtr> weights;
  size_t gpu_memory_bytes = 0U;
  bool has_weights_loaded = false;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
