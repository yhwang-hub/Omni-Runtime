#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_pipeline.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Omni-Runtime/inference/tokenizer/chat_template.h"
#include "Omni-Runtime/inference/tokenizer/gguf_tokenizer_loader.h"
#include "Omni-Runtime/inference/tokenizer/tokenizer_factory.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_decode_executor_gpu.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_prefill_executor_gpu.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_vision_encoder_executor_gpu.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {
namespace {

std::filesystem::path FindWorkspaceFile(const std::string &relative_path) {
  std::error_code error;
  std::filesystem::path directory = std::filesystem::current_path(error);
  while (!error) {
    const std::filesystem::path candidate = directory / relative_path;
    if (std::filesystem::is_regular_file(candidate, error)) {
      return candidate;
    }
    error.clear();
    const std::filesystem::path parent = directory.parent_path();
    if (parent == directory) {
      break;
    }
    directory = parent;
  }
  return std::filesystem::path(relative_path);
}

graph::NodeInfo MakeNodeInfo(const std::string &name,
                             const std::vector<std::shared_ptr<graph::Element>> &dependencies) {
  graph::NodeInfo info;
  info.name = name;
  info.dependencies = dependencies;
  return info;
}

class Qwen3VLTokenizeNode final : public graph::Node {
public:
  explicit Qwen3VLTokenizeNode(Qwen3VLGenerationContext *const context) : context_(context) {
  }

protected:
  graph::Status Run() final {
    OMNI_RETURN_VAL_IF_LOG(context_ == nullptr || context_->request == nullptr ||
                               context_->result == nullptr || context_->tokenizer == nullptr,
                           graph::InvalidArgumentStatus("Qwen3-VL tokenize context is null"), ERROR,
                           "Qwen3-VL tokenize context is null");
    Qwen3VLChatTemplate chat_template;
    OMNI_RETURN_VAL_IF_LOG(!chat_template.BuildUserPrompt(context_->request->prompt,
                                                          !context_->request->image_path.empty(),
                                                          &context_->model_prompt),
                           graph::InternalStatus("failed to build Qwen3-VL chat prompt"), ERROR,
                           "failed to build Qwen3-VL chat prompt");
    OMNI_RETURN_VAL_IF_LOG(
        !context_->tokenizer->Encode(context_->model_prompt, &context_->prompt_token_ids),
        graph::InternalStatus(context_->tokenizer->error_message()), ERROR,
        context_->tokenizer->error_message());
    return graph::Status();
  }

private:
  Qwen3VLGenerationContext *context_ = nullptr;
};

} // namespace

Qwen3VLPipeline::Qwen3VLPipeline() = default;

Qwen3VLPipeline::~Qwen3VLPipeline() {
  ResetGenerationState(runtime_.get());
}

bool Qwen3VLPipeline::Init(const std::string &config_path) {
  OMNI_RETURN_VAL_IF_LOG(is_generating_, false, ERROR,
                         "Cannot reinitialize the pipeline from a generation callback");
  error_message_.clear();
  is_initialized_ = false;
  OMNI_RETURN_VAL_IF(!Prepare(config_path) || !BuildGraph(), false);
  is_initialized_ = true;
  return true;
}

bool Qwen3VLPipeline::Prepare(const std::string &config_path) {
  OMNI_RETURN_VAL_IF(!config_.Load(config_path), SetError(config_.error_message()));
  OMNI_RETURN_VAL_IF(config_.model_name() != "qwen3_vl",
                     SetError("unsupported model_name: " + config_.model_name()));
  std::string model_path;
  std::string mmproj_path;
  OMNI_RETURN_VAL_IF(!config_.ResolveRelativePath(config_.model_path(), &model_path) ||
                         !config_.ResolveRelativePath(config_.mmproj_path(), &mmproj_path),
                     SetError("failed to resolve Qwen3-VL model paths"));
  OMNI_RETURN_VAL_IF(!model_serializer_.Parse(model_path),
                     SetError(model_serializer_.error_message()));
  OMNI_RETURN_VAL_IF(!vision_serializer_.Parse(mmproj_path),
                     SetError(vision_serializer_.error_message()));
  OMNI_RETURN_VAL_IF(!weights_.Build(model_serializer_, vision_serializer_),
                     SetError(weights_.error_message()));
  TokenizerModel tokenizer_model;
  GGUFTokenizerLoader tokenizer_loader;
  OMNI_RETURN_VAL_IF(!tokenizer_loader.Load(model_serializer_, &tokenizer_model),
                     SetError(tokenizer_loader.error_message()));
  tokenizer_ = TokenizerFactory::Create(tokenizer_model.type);
  OMNI_RETURN_VAL_IF(
      tokenizer_ == nullptr || !tokenizer_->Init(tokenizer_model),
      SetError(tokenizer_ == nullptr ? "failed to create tokenizer" : tokenizer_->error_message()));
  runtime_ = std::make_unique<Qwen3VLRuntime>();
  runtime_->config = &config_;
  runtime_->model_serializer = &model_serializer_;
  runtime_->vision_serializer = &vision_serializer_;
  runtime_->registry = &weights_;
  std::string runtime_error;
  OMNI_RETURN_VAL_IF(!InitializeRuntime(runtime_.get(), &runtime_error) ||
                         !LoadRuntimeWeights(runtime_.get(), &runtime_error),
                     SetError(runtime_error));
  return true;
}

bool Qwen3VLPipeline::BuildGraph() {
  OMNI_RETURN_VAL_IF(!inference_graph_.Init("qwen3_vl_inference_graph", 1U),
                     SetError(inference_graph_.error_message()));
  const auto tokenize_node = inference_graph_.RegisterNode<Qwen3VLTokenizeNode>(
      MakeNodeInfo("qwen3_vl_tokenize", {}), &context_);
  const auto vision_node = inference_graph_.RegisterNode<Qwen3VLVisionEncoderExecutorGpu>(
      MakeNodeInfo("qwen3_vl_vision", {}), runtime_.get(), &context_);
  const auto prefill_node = inference_graph_.RegisterNode<Qwen3VLPrefillExecutorGpu>(
      MakeNodeInfo("qwen3_vl_prefill", {tokenize_node, vision_node}), runtime_.get(), &context_);
  const auto decode_node = inference_graph_.RegisterNode<Qwen3VLDecodeExecutorGpu>(
      MakeNodeInfo("qwen3_vl_decode", {prefill_node}), runtime_.get(), &context_);
  OMNI_RETURN_VAL_IF(tokenize_node == nullptr || vision_node == nullptr ||
                         prefill_node == nullptr || decode_node == nullptr ||
                         !vision_node->Initialize() || !prefill_node->Initialize() ||
                         !decode_node->Initialize(),
                     SetError("failed to register or initialize Qwen3-VL graph nodes"));
  OMNI_RETURN_VAL_IF(!inference_graph_.Prepare(), SetError(inference_graph_.error_message()));
  return true;
}

bool Qwen3VLPipeline::Generate(const GenerationRequest &request, GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(result == nullptr, SetError("generation result is null"));
  OMNI_RETURN_VAL_IF_LOG(is_generating_, false, ERROR,
                         "Cannot reenter generation from a stream callback");
  OMNI_RETURN_VAL_IF(!is_initialized_ || runtime_ == nullptr || tokenizer_ == nullptr,
                     SetError("Qwen3-VL pipeline is not initialized"));
  OMNI_RETURN_VAL_IF(request.prompt.empty(), SetError("generation prompt is empty"));
  OMNI_RETURN_VAL_IF(request.max_new_tokens <= 0, SetError("generation max_new_tokens is invalid"));
  OMNI_RETURN_VAL_IF(!request.image_path.empty() &&
                         !std::filesystem::is_regular_file(request.image_path),
                     SetError("generation image does not exist"));
  *result = GenerationResult{};
  error_message_.clear();
  OMNI_RETURN_VAL_IF(!ResetGenerationState(runtime_.get()), SetError("failed to reset generation"));
  context_ = Qwen3VLGenerationContext{};
  context_.request = &request;
  context_.result = result;
  context_.tokenizer = tokenizer_.get();
  context_.model_config = &weights_.model_config();
  is_generating_ = true;
  const bool is_processed = inference_graph_.Process();
  is_generating_ = false;
  const std::string graph_error = inference_graph_.error_message();
  const std::string context_error = context_.error_message;
  context_.request = nullptr;
  context_.result = nullptr;
  OMNI_RETURN_VAL_IF(!is_processed, SetError(context_error.empty() ? graph_error : context_error));
  return true;
}

bool Qwen3VLPipeline::Run(std::string *const output_text) {
  OMNI_RETURN_VAL_IF(output_text == nullptr, SetError("output text is null"));
  GenerationRequest request;
  request.prompt = "Describe this image.";
  request.image_path = FindWorkspaceFile("test/inference/demo.jpeg").string();
  request.max_new_tokens = config_.max_new_tokens();
  GenerationResult result;
  OMNI_RETURN_VAL_IF(!Generate(request, &result), false);
  *output_text = result.text;
  return true;
}

bool Qwen3VLPipeline::SetError(const std::string &message) {
  error_message_ = message;
  return false;
}

const Qwen3VLConfig &Qwen3VLPipeline::model_config() const {
  return weights_.model_config();
}
const std::string &Qwen3VLPipeline::error_message() const {
  return error_message_;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
