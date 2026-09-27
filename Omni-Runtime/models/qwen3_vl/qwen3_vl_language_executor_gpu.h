#pragma once

#include <cstdint>
#include <vector>

#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

struct Qwen3VLRuntime;

bool RunQwen3VLLanguageStep(Qwen3VLRuntime *const runtime, const std::vector<int32_t> &token_ids,
                            const std::vector<int32_t> &position_ids,
                            const Qwen3VLVisionFeatures &vision_features,
                            const int32_t vision_token_offset, int32_t *const next_token_id);

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
