#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"
#include "Omni-Runtime/models/qwen3_vl/config_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

class Qwen3VLWeightSchema final {
public:
  bool Build(const util::GGUFSerializer &model_serializer,
             const util::GGUFSerializer &vision_serializer);

  const Qwen3VLConfig &model_config() const;
  const qwen3::Qwen3VisionConfig &vision_config() const;
  const std::string &output_weight_name() const;
  const std::vector<Qwen3VLWeightName> &model_weights() const;
  const std::vector<Qwen3VLWeightName> &vision_weights() const;
  const std::string &error_message() const;

private:
  bool BuildModelRegistry(const util::GGUFSerializer &serializer, const Qwen3VLConfig &config);
  bool BuildVisionRegistry(const util::GGUFSerializer &serializer);
  bool ValidateTensor(const util::GGUFSerializer &serializer, const Qwen3VLWeightName &weight);
  bool SetError(const std::string &message);

  Qwen3VLConfig model_config_;
  qwen3::Qwen3VisionConfig vision_config_;
  std::string output_weight_name_;
  std::vector<Qwen3VLWeightName> model_weights_;
  std::vector<Qwen3VLWeightName> vision_weights_;
  std::string error_message_;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
