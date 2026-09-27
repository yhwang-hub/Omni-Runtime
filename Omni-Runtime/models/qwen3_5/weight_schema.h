#pragma once

#include <string>
#include <unordered_map>

#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"
#include "Omni-Runtime/inference/model_loader/weight_define.h"
#include "Omni-Runtime/inference/tokenizer/tokenizer_define.h"
#include "Omni-Runtime/models/qwen3_5/config_define.h"
#include "Omni-Runtime/models/components/qwen_vision/config.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_5 {

class Qwen35WeightSchema final {
public:
  bool Build(const util::GGUFSerializer &model, const util::GGUFSerializer &vision,
             const TokenizerModel &tokenizer);

  const Qwen35Config &model_config() const {
    return model_config_;
  }

  const std::unordered_map<std::string, WeightStorage> &weights() const {
    return weights_;
  }

  const std::string &error_message() const {
    return error_message_;
  }

private:
  bool LoadMetadata(const util::GGUFSerializer &model, const util::GGUFSerializer &vision);
  bool RegisterWeights(const util::GGUFSerializer &serializer);
  bool ValidateWeights() const;
  bool SetError(const std::string &message) {
    error_message_ = message;
    return false;
  }

  Qwen35Config model_config_;
  std::unordered_map<std::string, WeightStorage> weights_;
  std::string error_message_;
};

} // namespace qwen3_5
} // namespace inference
} // namespace omni_runtime
