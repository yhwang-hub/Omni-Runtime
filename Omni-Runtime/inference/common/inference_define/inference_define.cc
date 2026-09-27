#include "Omni-Runtime/inference/common/inference_define/inference_define.h"

namespace omni_runtime {
namespace inference {

const char *ModelBackendName(const ModelBackend backend) {
  switch (backend) {
  case ModelBackend::kAuto:
    return "auto";
  case ModelBackend::kCpu:
    return "cpu";
  case ModelBackend::kGpu:
    return "gpu";
  }
  return "unknown";
}

const char *ModelWeightTypeName(const ModelWeightType weight_type) {
  switch (weight_type) {
  case ModelWeightType::kAuto:
    return "auto";
  case ModelWeightType::kFloat32:
    return "float32";
  case ModelWeightType::kFloat16:
    return "float16";
  case ModelWeightType::kBFloat16:
    return "bfloat16";
  case ModelWeightType::kInt8:
    return "int8";
  }
  return "unknown";
}

} // namespace inference
} // namespace omni_runtime
