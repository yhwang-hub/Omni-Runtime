#pragma once

#include <memory>
#include <string>

#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"
#include "Omni-Runtime/inference/common/inference_config/inference_config.h"
#include "Omni-Runtime/inference/inference_graph/inference_graph.h"
#include "Omni-Runtime/inference/tokenizer/tokenizer.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_define.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_runtime.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_weights.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

class Qwen3VLPipeline final {
public:
  Qwen3VLPipeline();
  ~Qwen3VLPipeline();
  Qwen3VLPipeline(const Qwen3VLPipeline &) = delete;
  Qwen3VLPipeline &operator=(const Qwen3VLPipeline &) = delete;

  bool Init(const std::string &config_path);
  bool Generate(const GenerationRequest &request, GenerationResult *const result);
  bool Run(std::string *const output_text);
  const Qwen3VLConfig &model_config() const;
  const std::string &error_message() const;

private:
  bool Prepare(const std::string &config_path);
  bool BuildGraph();
  bool SetError(const std::string &message);

  InferenceConfig config_;
  util::GGUFSerializer model_serializer_;
  util::GGUFSerializer vision_serializer_;
  Qwen3VLWeightRegistry weights_;
  Qwen3VLGenerationContext context_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Qwen3VLRuntime> runtime_;
  InferenceGraph inference_graph_;
  bool is_initialized_ = false;
  bool is_generating_ = false;
  std::string error_message_;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
