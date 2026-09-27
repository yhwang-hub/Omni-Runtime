#include "Omni-Runtime/models/registry.h"

#include "Omni-Runtime/models/qwen3_vl/model.h"
#include "Omni-Runtime/models/qwen3_5/model.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {

bool RegisterBuiltinModels(ModelRegistry *const registry) {
  OMNI_RETURN_VAL_IF_LOG(registry == nullptr, false, ERROR, "Model registry is null");
  const ModelRegistry::Factory qwen_vl_factory = []() {
    return std::make_unique<qwen3_vl::Qwen3VLModel>();
  };
  const ModelRegistry::Factory qwen35 = []() { return std::make_unique<qwen3_5::Qwen35Model>(); };
  OMNI_RETURN_VAL_IF(!registry->Register("qwen3_vl", qwen_vl_factory) ||
                         !registry->Register("qwen3vl", qwen_vl_factory) ||
                         !registry->Register("qwen3_8_nvfp4", qwen35) ||
                         !registry->Register("qwen3_8_fp8", qwen35) ||
                         !registry->Register("qwen3_5", qwen35) ||
                         !registry->Register("qwen35", qwen35),
                     false);
  return true;
}

} // namespace inference
} // namespace omni_runtime
