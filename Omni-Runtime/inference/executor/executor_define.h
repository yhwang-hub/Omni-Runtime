#pragma once

#include <string>
#include <variant>
#include <vector>

#include "Omni-Runtime/inference/common/inference_define/inference_define.h"
#include "Omni-Runtime/memory/tensor.h"

namespace omni_runtime {
namespace inference {

struct ActionRequest final {
  std::vector<memory::TensorPtr> observations;
  std::vector<float> state;
  int32_t horizon = 0;
};

struct ActionResult final {
  std::vector<float> actions;
  int32_t horizon = 0;
  int32_t action_size = 0;
};

using ModelRequest = std::variant<GenerationRequest, ActionRequest>;
using ModelResult = std::variant<GenerationResult, ActionResult>;

} // namespace inference
} // namespace omni_runtime
