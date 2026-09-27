#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/inference/executor/model_executor.h"
#include "Omni-Runtime/inference/inference_graph/operator_graph.h"
#include "Omni-Runtime/memory/buffer.h"
#include "Omni-Runtime/models/components/qwen_multimodal/processor_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

class Qwen3VLModel;

class Qwen3VLExecutor final : public ModelExecutor {
public:
  explicit Qwen3VLExecutor(const Qwen3VLModel &model);
  bool Init() override;
  bool Execute(const ModelRequest &request, ModelResult *const result) override;

private:
  struct LanguageWorkspaces final {
    memory::BufferPtr tokens;
    memory::BufferPtr positions;
    memory::BufferPtr hidden;
    memory::BufferPtr next_hidden;
    memory::BufferPtr normalized;
    memory::BufferPtr query;
    memory::BufferPtr key;
    memory::BufferPtr value;
    memory::BufferPtr query_normed;
    memory::BufferPtr key_normed;
    memory::BufferPtr attention;
    memory::BufferPtr projected;
    memory::BufferPtr gate;
    memory::BufferPtr up;
    memory::BufferPtr logits;
    memory::BufferPtr sample;
    memory::BufferPtr decode_partials;
  };

  struct VisionWorkspaces final {
    memory::BufferPtr source;
    memory::BufferPtr patches;
    memory::BufferPtr patch_embedding;
    memory::BufferPtr hidden;
    memory::BufferPtr residual;
    memory::BufferPtr normalized;
    memory::BufferPtr qkv;
    memory::BufferPtr query;
    memory::BufferPtr key;
    memory::BufferPtr value;
    memory::BufferPtr attention;
    memory::BufferPtr projected;
    memory::BufferPtr feed_forward;
    memory::BufferPtr embedding;
    std::vector<memory::BufferPtr> deepstack;
  };

  struct RequestState final {
    memory::BufferPtr key_cache;
    memory::BufferPtr value_cache;
    memory::BufferPtr decode_cache_position;
    int32_t cache_capacity = 0;
    int32_t used_token_count = 0;
    int32_t next_rope_position = 0;
  };

  struct ExecutionContext final {
    size_t gpu_memory_bytes = 0U;
    backend::EventTimer compute_timer;
    double step_time_ms = 0.0;
    LanguageWorkspaces language_workspaces;
    VisionWorkspaces vision_workspaces;
    OperatorGraph embedding_graph;
    OperatorGraph language_graph;
    OperatorGraph vision_graph;
    std::unique_ptr<backend::ExecutionPlan> decode_plan;
  };

  struct VisionFeatures final {
    memory::BufferPtr embedding;
    std::vector<memory::BufferPtr> deepstack_embeddings;
    ImageSize input_image_size;
    ImageSize resized_image_size;
    int32_t token_count = 0;
    int32_t merged_grid_height = 0;
    int32_t merged_grid_width = 0;
  };

  struct GenerationContext final : public qwen3::Prompt {
    VisionFeatures vision_features;
    int32_t next_token_id = -1;
  };

  bool RunDecodeStep(int32_t *const token, double *const elapsed_ms);
  bool AllocateKvCache(const int32_t capacity);
  bool AllocateWorkspaces();
  bool ResetState();
  backend::ScalarType RuntimeScalarType() const;
  size_t RuntimeElementSize() const;
  backend::DeviceView View(const std::string &name) const;
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
  const Qwen3VLModel &model_;
  OperatorGraphConfig graph_config_;
  RequestState state_;
  std::unique_ptr<ExecutionContext> context_;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
