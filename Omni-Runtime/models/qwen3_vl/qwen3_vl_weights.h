#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

class Qwen3VLWeightRegistry final {
public:
  bool Build(const util::GGUFSerializer &model_serializer,
             const util::GGUFSerializer &vision_serializer);

  const Qwen3VLConfig &model_config() const;
  const Qwen3VLVisionConfig &vision_config() const;
  const Qwen3VLWeightName &token_embedding() const;
  const Qwen3VLWeightName &output_norm() const;
  const Qwen3VLWeightName &output_weight() const;
  const std::vector<Qwen3VLLayerWeightNames> &layers() const;
  const std::vector<Qwen3VLWeightName> &vision_weights() const;
  bool has_output_weight() const;
  const std::string &error_message() const;

private:
  bool BuildModelRegistry(const util::GGUFSerializer &serializer, const Qwen3VLConfig &config);
  bool BuildVisionRegistry(const util::GGUFSerializer &serializer);
  bool ValidateTensor(const util::GGUFSerializer &serializer, const std::string &name,
                      const std::vector<uint64_t> &dimensions, Qwen3VLWeightName *const output);
  bool SetError(const std::string &message);

  Qwen3VLConfig model_config_;
  Qwen3VLVisionConfig vision_config_;
  Qwen3VLWeightName token_embedding_;
  Qwen3VLWeightName output_norm_;
  Qwen3VLWeightName output_weight_;
  std::vector<Qwen3VLLayerWeightNames> layers_;
  std::vector<Qwen3VLWeightName> vision_weights_;
  bool has_output_weight_ = false;
  std::string error_message_;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
