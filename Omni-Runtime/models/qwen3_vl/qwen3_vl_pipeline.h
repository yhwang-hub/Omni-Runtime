#pragma once

#include <string>

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

class Qwen3VLPipeline final {
public:
  bool Init(const std::string &config_path);
  bool Run(std::string *const output_text);
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
