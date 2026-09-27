#pragma once

#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"
#include "Omni-Runtime/models/components/qwen_vision/config_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3 {

bool LoadVisionMetadata(const util::GGUFSerializer &serializer, Qwen3VisionConfig *const config);

} // namespace qwen3
} // namespace inference
} // namespace omni_runtime
