#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/memory/buffer.h"
#include "Omni-Runtime/inference/executor/model_executor.h"
#include "Omni-Runtime/inference/inference_graph/operator_graph.h"
#include "Omni-Runtime/models/components/qwen_multimodal/processor_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_5 {

class Qwen35Model;

class Qwen35Executor final : public ModelExecutor {
public:
  explicit Qwen35Executor(const Qwen35Model &model);
  bool Init() override;
  bool Execute(const ModelRequest &request, ModelResult *const result) override;

private:
  struct LayerState final {
    memory::BufferPtr key;
    memory::BufferPtr value;
    memory::BufferPtr convolution;
    memory::BufferPtr recurrent;
  };

  struct RequestState final {
    std::vector<LayerState> layers;
  };

  struct ExecutionContext final {
    std::shared_ptr<backend::Backend> backend;
    std::unique_ptr<backend::EventTimer> timer;
    std::unordered_map<std::string, memory::BufferPtr> buffers;
    size_t gpu_memory_bytes = 0U;
    std::unique_ptr<backend::ExecutionPlan> decode_plan;
  };

  bool RunDecodeStep(int32_t *const token, double *const elapsed_ms);
  bool Allocate(const std::string &name, const size_t bytes);
  bool AllocateStorage();
  bool ResetState();
  bool EncodeImage(const GenerationRequest &request, qwen3::Prompt *const prompt,
                   GenerationResult *const result);
  bool Prefill(const qwen3::Prompt &prompt, int32_t *const next_token,
               GenerationResult *const result);
  bool MakeLanguageNodes(const int32_t tokens, const int32_t start, const bool is_decode,
                         std::vector<backend::ComputeNode> *const embedding,
                         std::vector<backend::ComputeNode> *const nodes);
  bool PrepareDecode();
  bool BindLayer(const int32_t layer, OperatorGraphBindings *const bindings) const;
  OperatorGraphBindings Bindings() const;
  backend::DeviceView View(const std::string &name) const;
  backend::ScalarType ComputeType() const;

  static constexpr int32_t kPrefillBatchTokens = 512;
  const Qwen35Model &model_;
  OperatorGraphConfig graph_config_;
  RequestState state_;
  ExecutionContext context_;
};

} // namespace qwen3_5
} // namespace inference
} // namespace omni_runtime
