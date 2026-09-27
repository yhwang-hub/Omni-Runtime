#include "Omni-Runtime/models/components/qwen_vision/config.h"

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3 {

bool LoadVisionMetadata(const util::GGUFSerializer &serializer, Qwen3VisionConfig *const config) {
  OMNI_RETURN_VAL_IF_LOG(config == nullptr, false, ERROR, "Vision metadata output is null");
  const std::pair<std::string, int32_t *> integers[] = {
      {"image_size", &config->image_size},
      {"patch_size", &config->patch_size},
      {"embedding_length", &config->hidden_size},
      {"feed_forward_length", &config->feed_forward_size},
      {"block_count", &config->layer_count},
      {"attention.head_count", &config->head_count},
      {"projection_dim", &config->projection_size},
      {"spatial_merge_size", &config->merge_size},
  };
  for (const auto &[key, output] : integers) {
    OMNI_RETURN_VAL_IF(!serializer.GetIntegerMetadata("clip.vision." + key, output), false);
  }
  return serializer.GetMetadata("clip.vision.attention.layer_norm_epsilon",
                                &config->layer_norm_epsilon);
}

} // namespace qwen3
} // namespace inference
} // namespace omni_runtime
