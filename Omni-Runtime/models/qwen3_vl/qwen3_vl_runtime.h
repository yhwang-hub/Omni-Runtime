#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/inference/common/inference_config/inference_config.h"
#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"
#include "Omni-Runtime/memory/buffer.h"
#include "Omni-Runtime/memory/tensor.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_define.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_weights.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

struct Qwen3VLRuntime final {
  std::shared_ptr<backend::Backend> backend;
  backend::EventTimer compute_timer;
  double step_time_ms = 0.0;
  const InferenceConfig *config = nullptr;
  const util::GGUFSerializer *model_serializer = nullptr;
  const util::GGUFSerializer *vision_serializer = nullptr;
  const Qwen3VLWeightRegistry *registry = nullptr;
  Qwen3VLConfig model_config;
  Qwen3VLVisionConfig vision_config;
  std::unordered_map<std::string, memory::TensorPtr> weights;
  Qwen3VLLanguageWorkspaces language_workspaces;
  Qwen3VLVisionWorkspaces vision_workspaces;
  memory::BufferPtr key_cache;
  memory::BufferPtr value_cache;
  int32_t cache_capacity = 0;
  int32_t used_token_count = 0;
  int32_t next_rope_position = 0;
  size_t gpu_memory_bytes = 0U;
  bool has_weights_loaded = false;
};

bool InitializeRuntime(Qwen3VLRuntime *const runtime, std::string *const error_message);
bool LoadRuntimeWeights(Qwen3VLRuntime *const runtime, std::string *const error_message);
bool AllocateKvCache(const int32_t capacity, Qwen3VLRuntime *const runtime);
bool AllocateWorkspaces(Qwen3VLRuntime *const runtime);
backend::ScalarType RuntimeScalarType(const Qwen3VLRuntime &runtime);
size_t RuntimeElementSize(const Qwen3VLRuntime &runtime);
const void *Weight(const Qwen3VLRuntime &runtime, const std::string &name);
bool LaunchMatmul(Qwen3VLRuntime *const runtime, const void *const input,
                  const std::string &weight_name, const int32_t rows, const int32_t inner,
                  const int32_t columns, void *const output);
bool ResetGenerationState(Qwen3VLRuntime *const runtime);

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
