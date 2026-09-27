#include "Omni-Runtime/inference/common/inference_config/inference_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace {

std::string Trim(const std::string &source) {
  const auto is_space = [](const unsigned char character) { return std::isspace(character) != 0; };
  const auto first = std::find_if_not(source.begin(), source.end(), is_space);
  const auto last = std::find_if_not(source.rbegin(), source.rend(), is_space).base();
  return first < last ? std::string(first, last) : std::string();
}

template <typename T> bool ParseNumber(const std::string &source, T *const output) {
  OMNI_RETURN_VAL_IF(output == nullptr, false);
  std::istringstream stream(source);
  T value = T{};
  stream >> value;
  OMNI_RETURN_VAL_IF(!stream || !stream.eof(), false);
  *output = value;
  return true;
}

bool ParseWeightType(const std::string &source, ModelWeightType *const output) {
  OMNI_RETURN_VAL_IF(output == nullptr, false);
  if (source == "auto") {
    *output = ModelWeightType::kAuto;
  } else if (source == "float32" || source == "fp32" || source == "f32") {
    *output = ModelWeightType::kFloat32;
  } else if (source == "float16" || source == "fp16" || source == "f16") {
    *output = ModelWeightType::kFloat16;
  } else if (source == "bfloat16" || source == "bf16") {
    *output = ModelWeightType::kBFloat16;
  } else if (source == "int8" || source == "q8_0") {
    *output = ModelWeightType::kInt8;
  } else {
    return false;
  }
  return true;
}

} // namespace

bool InferenceConfig::Load(const std::string &file_path) {
  error_message_.clear();
  graph_path_.clear();
  OMNI_RETURN_VAL_IF(file_path.empty(), SetError("inference config path is empty"));
  std::ifstream input(file_path);
  OMNI_RETURN_VAL_IF(!input.is_open(), SetError("failed to open inference config: " + file_path));
  config_directory_ = std::filesystem::path(file_path).parent_path();
  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed_line = Trim(line);
    if (trimmed_line.empty() || trimmed_line.front() == '#') {
      continue;
    }
    const size_t separator = trimmed_line.find('=');
    OMNI_RETURN_VAL_IF(separator == std::string::npos || separator == 0U ||
                           separator + 1U >= trimmed_line.size(),
                       SetError("invalid inference config line: " + line));
    OMNI_RETURN_VAL_IF(!SetValue(Trim(trimmed_line.substr(0U, separator)),
                                 Trim(trimmed_line.substr(separator + 1U))),
                       false);
  }
  OMNI_RETURN_VAL_IF(model_name_.empty(), SetError("model_name is missing"));
  return true;
}

bool InferenceConfig::ResolveRelativePath(const std::string &relative_path,
                                          std::string *const resolved_path) const {
  OMNI_RETURN_VAL_IF(resolved_path == nullptr, false);
  const std::filesystem::path path(relative_path);
  if (path.is_absolute() && std::filesystem::is_regular_file(path)) {
    *resolved_path = relative_path;
    return true;
  }
  const std::filesystem::path config_path = config_directory_ / path;
  if (std::filesystem::is_regular_file(config_path)) {
    *resolved_path = std::filesystem::absolute(config_path).string();
    return true;
  }
  const std::filesystem::path runfiles_path =
      config_directory_.parent_path().parent_path().parent_path() / path;
  if (std::filesystem::is_regular_file(runfiles_path)) {
    *resolved_path = std::filesystem::absolute(runfiles_path).string();
    return true;
  }
  OMNI_RETURN_VAL_IF(!std::filesystem::is_regular_file(path), false);
  *resolved_path = std::filesystem::absolute(path).string();
  return true;
}

bool InferenceConfig::SetValue(const std::string &key, const std::string &value) {
  if (key == "model_name") {
    model_name_ = value;
  } else if (key == "runtime_backend") {
    OMNI_RETURN_VAL_IF(value != "gpu", SetError("unsupported runtime_backend: " + value));
    model_backend_ = ModelBackend::kGpu;
  } else if (key == "compute_backend") {
    compute_backend_ = value;
  } else if (key == "model_path") {
    model_path_ = value;
  } else if (key == "mmproj_path") {
    mmproj_path_ = value;
  } else if (key == "graph_path") {
    graph_path_ = value;
  } else if (key == "model_weight_type") {
    OMNI_RETURN_VAL_IF(!ParseWeightType(value, &model_weight_type_),
                       SetError("invalid model_weight_type: " + value));
  } else if (key == "compute_dtype") {
    OMNI_RETURN_VAL_IF(!ParseWeightType(value, &compute_type_),
                       SetError("invalid compute_dtype: " + value));
  } else if (key == "max_context_tokens") {
    OMNI_RETURN_VAL_IF(!ParseNumber(value, &max_context_tokens_),
                       SetError("invalid max_context_tokens: " + value));
  } else if (key == "max_new_tokens") {
    OMNI_RETURN_VAL_IF(!ParseNumber(value, &max_new_tokens_),
                       SetError("invalid max_new_tokens: " + value));
  } else if (key == "gpu_layer_count") {
    OMNI_RETURN_VAL_IF(!ParseNumber(value, &gpu_layer_count_),
                       SetError("invalid gpu_layer_count: " + value));
  } else if (key == "image_min_tokens") {
    OMNI_RETURN_VAL_IF(!ParseNumber(value, &image_min_tokens_),
                       SetError("invalid image_min_tokens: " + value));
  }
  return true;
}

bool InferenceConfig::SetError(const std::string &message) {
  error_message_ = message;
  return false;
}

const std::string &InferenceConfig::model_name() const {
  return model_name_;
}
ModelBackend InferenceConfig::model_backend() const {
  return model_backend_;
}
const std::string &InferenceConfig::compute_backend() const {
  return compute_backend_;
}
void InferenceConfig::set_compute_backend(const std::string &compute_backend) {
  compute_backend_ = compute_backend;
}
const std::string &InferenceConfig::model_path() const {
  return model_path_;
}
const std::string &InferenceConfig::mmproj_path() const {
  return mmproj_path_;
}
const std::string &InferenceConfig::graph_path() const {
  return graph_path_;
}
ModelWeightType InferenceConfig::model_weight_type() const {
  return model_weight_type_;
}
ModelWeightType InferenceConfig::compute_type() const {
  return compute_type_;
}
int32_t InferenceConfig::max_context_tokens() const {
  return max_context_tokens_;
}
int32_t InferenceConfig::max_new_tokens() const {
  return max_new_tokens_;
}
int32_t InferenceConfig::gpu_layer_count() const {
  return gpu_layer_count_;
}
int32_t InferenceConfig::image_min_tokens() const {
  return image_min_tokens_;
}
const std::string &InferenceConfig::error_message() const {
  return error_message_;
}

} // namespace inference
} // namespace omni_runtime
