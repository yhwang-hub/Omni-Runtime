#pragma once

#include <string>
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

bool InitializeRuntime(Qwen3VLRuntime *const runtime, std::string *const error_message);
bool LoadRuntimeWeights(Qwen3VLRuntime *const runtime, std::string *const error_message);
bool AllocateKvCache(const int32_t capacity, Qwen3VLRuntime *const runtime);
bool AllocateWorkspaces(Qwen3VLRuntime *const runtime);
backend::ScalarType RuntimeScalarType(const Qwen3VLRuntime &runtime);
size_t RuntimeElementSize(const Qwen3VLRuntime &runtime);
backend::DeviceView Weight(const Qwen3VLRuntime &runtime, const std::string &name);
bool ResetGenerationState(Qwen3VLRuntime *const runtime);

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
