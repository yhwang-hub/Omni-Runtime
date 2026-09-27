#pragma once

#include "Omni-Runtime/inference/common/image_preprocessor/image_preprocessor_define.h"
#include "Omni-Runtime/inference/inference_graph/operator_graph_define.h"
#include "Omni-Runtime/models/components/qwen_vision/config_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3 {

bool BindVisionDimensions(const Qwen3VisionConfig &config, const ImagePatchGrid &grid,
                          const int32_t width_pixels, const int32_t height_pixels,
                          OperatorGraphBindings *const bindings);

} // namespace qwen3
} // namespace inference
} // namespace omni_runtime
