#pragma once

#include <memory>
#include <string>

#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"
#include "Omni-Runtime/inference/common/inference_config/inference_config.h"
#include "Omni-Runtime/inference/inference_graph/operator_graph.h"
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
  const std::string &compute_backend() const;
  const std::string &error_message() const;

private:
  struct GenerationContext final {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> position_ids;
    Qwen3VLVisionFeatures vision_features;
    int32_t vision_token_offset = -1;
    int32_t next_token_id = -1;
  };

  bool Prepare(const std::string &config_path);
  bool BuildGraph();
  bool RunVision(const GenerationRequest &request, GenerationContext *const context,
                 GenerationResult *const result);
  bool PreparePrompt(const GenerationRequest &request, GenerationContext *const context);
  bool RunPrefill(GenerationContext *const context, GenerationResult *const result);
  bool RunDecode(const GenerationRequest &request, const int32_t first_token_id,
                 GenerationResult *const result);
  bool BuildLanguageBindings(const int32_t token_count, const bool is_decode,
                             const GenerationContext &context,
                             OperatorGraphBindings *const bindings) const;
  bool BindLanguageLayer(const std::string &variable, const int32_t layer,
                         const GenerationContext &context, const bool is_decode,
                         OperatorGraphBindings *const bindings) const;
  OperatorGraphBindings MakeBindings() const;
  bool SetError(const std::string &message);

  InferenceConfig config_;
  util::GGUFSerializer model_serializer_;
  util::GGUFSerializer vision_serializer_;
  Qwen3VLWeightRegistry weights_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Qwen3VLRuntime> runtime_;
  OperatorGraphConfig graph_config_;
  OperatorGraph embedding_graph_;
  OperatorGraph language_graph_;
  OperatorGraph vision_graph_;
  bool is_initialized_ = false;
  bool is_generating_ = false;
  std::string error_message_;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
