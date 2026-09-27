#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "Omni-Runtime/inference/common/inference_define/inference_define.h"

namespace omni_runtime {
namespace inference {

class InferenceConfig final {
public:
  bool Load(const std::string &file_path);
  bool ResolveRelativePath(const std::string &relative_path,
                           std::string *const resolved_path) const;

  const std::string &model_name() const;
  ModelBackend model_backend() const;
  const std::string &model_path() const;
  const std::string &mmproj_path() const;
  ModelWeightType model_weight_type() const;
  ModelWeightType compute_type() const;
  int32_t max_context_tokens() const;
  int32_t max_new_tokens() const;
  int32_t gpu_layer_count() const;
  int32_t image_min_tokens() const;
  const std::string &error_message() const;

private:
  bool SetValue(const std::string &key, const std::string &value);
  bool SetError(const std::string &message);

  std::string model_name_;
  ModelBackend model_backend_ = ModelBackend::kAuto;
  std::string model_path_;
  std::string mmproj_path_;
  ModelWeightType model_weight_type_ = ModelWeightType::kAuto;
  ModelWeightType compute_type_ = ModelWeightType::kFloat32;
  int32_t max_context_tokens_ = 16384;
  int32_t max_new_tokens_ = 64;
  int32_t gpu_layer_count_ = 99;
  int32_t image_min_tokens_ = 1024;
  std::filesystem::path config_directory_;
  std::string error_message_;
};

} // namespace inference
} // namespace omni_runtime
