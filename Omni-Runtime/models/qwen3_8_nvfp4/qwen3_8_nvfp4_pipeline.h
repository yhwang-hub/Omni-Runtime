#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"
#include "Omni-Runtime/inference/common/inference_config/inference_config.h"
#include "Omni-Runtime/inference/inference_graph/operator_graph.h"
#include "Omni-Runtime/inference/tokenizer/tokenizer.h"
#include "Omni-Runtime/memory/buffer.h"
#include "Omni-Runtime/models/qwen3_8_nvfp4/qwen3_8_nvfp4_define.h"

namespace omni_runtime::inference::qwen3_8_nvfp4 {

class Qwen38Pipeline final {
public:
  Qwen38Pipeline();
  ~Qwen38Pipeline();
  Qwen38Pipeline(const Qwen38Pipeline &) = delete;
  Qwen38Pipeline &operator=(const Qwen38Pipeline &) = delete;

  bool Init(const std::string &config_path);
  bool Generate(const GenerationRequest &request, GenerationResult *const result);
  const Qwen38Config &model_config() const;
  const std::string &compute_backend() const;
  const std::string &error_message() const;

private:
  struct Weight final {
    memory::BufferPtr storage;
    std::vector<uint64_t> dimensions;
    int32_t type = 0;
    float scale = 1.0F;
  };

  struct LayerState final {
    memory::BufferPtr key;
    memory::BufferPtr value;
    memory::BufferPtr convolution;
    memory::BufferPtr recurrent;
  };

  struct Prompt final {
    std::vector<int32_t> tokens;
    std::vector<int32_t> positions;
    int32_t image_offset = -1;
    int32_t image_tokens = 0;
    int32_t image_height = 0;
    int32_t image_width = 0;
    int32_t next_position = 0;
  };

  bool LoadMetadata(const util::GGUFSerializer &model, const util::GGUFSerializer &vision);
  bool LoadWeights(const util::GGUFSerializer &serializer);
  bool ValidateWeights() const;
  bool Allocate(const std::string &name, const size_t bytes);
  bool AllocateStorage();
  bool ResetState();
  bool EncodeImage(const GenerationRequest &request, Prompt *const prompt,
                   GenerationResult *const result);
  bool PreparePrompt(const GenerationRequest &request, Prompt *const prompt);
  bool Prefill(const Prompt &prompt, int32_t *const next_token, GenerationResult *const result);
  bool Decode(const GenerationRequest &request, const int32_t first_token,
              GenerationResult *const result);
  bool MakeLanguageNodes(const int32_t tokens, const int32_t start, const bool is_decode,
                         std::vector<backend::ComputeNode> *const embedding,
                         std::vector<backend::ComputeNode> *const nodes);
  bool PrepareDecode();
  bool BindLayer(const int32_t layer, OperatorGraphBindings *const bindings) const;
  OperatorGraphBindings Bindings() const;
  backend::DeviceView View(const std::string &name) const;
  bool SetError(const std::string &message);

  static constexpr int32_t kPrefillBatchTokens = 512;
  InferenceConfig inference_config_;
  Qwen38Config model_config_;
  OperatorGraphConfig graph_config_;
  std::unique_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<backend::Backend> backend_;
  std::unique_ptr<backend::EventTimer> timer_;
  std::unordered_map<std::string, Weight> weights_;
  std::unordered_map<std::string, memory::BufferPtr> buffers_;
  std::vector<LayerState> states_;
  std::unique_ptr<backend::ExecutionPlan> decode_plan_;
  size_t gpu_memory_bytes_ = 0U;
  bool is_initialized_ = false;
  bool is_generating_ = false;
  std::string error_message_;
};

} // namespace omni_runtime::inference::qwen3_8_nvfp4
