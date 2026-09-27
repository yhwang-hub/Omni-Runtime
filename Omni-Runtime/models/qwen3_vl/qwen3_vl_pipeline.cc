#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_pipeline.h"

#include <filesystem>
#include <memory>
#include <string>

#include "Omni-Runtime/inference/tokenizer/gguf_tokenizer_loader.h"
#include "Omni-Runtime/inference/tokenizer/tokenizer_factory.h"
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

} // namespace

Qwen3VLPipeline::Qwen3VLPipeline() = default;

Qwen3VLPipeline::~Qwen3VLPipeline() = default;

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
  config_ = InferenceConfig{};
  OMNI_RETURN_VAL_IF(!config_.Load(config_path), SetError(config_.error_message()));
  OMNI_RETURN_VAL_IF(config_.model_name() != "qwen3_vl",
                     SetError("unsupported model_name: " + config_.model_name()));
  OMNI_RETURN_VAL_IF(config_.graph_path().empty(), SetError("Qwen3-VL graph_path is missing"));
  std::string graph_path;
  OMNI_RETURN_VAL_IF(!config_.ResolveRelativePath(config_.graph_path(), &graph_path),
                     SetError("failed to resolve Qwen3-VL graph_path: " + config_.graph_path()));
  OMNI_RETURN_VAL_IF(!graph_config_.Load(graph_path), SetError(graph_config_.error_message()));
  config_.set_compute_backend(graph_config_.backend());
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
  GenerationContext context;
  is_generating_ = true;
  const bool is_processed = RunVision(request, &context, result) &&
                            PreparePrompt(request, &context) && RunPrefill(&context, result) &&
                            RunDecode(request, context.next_token_id, result);
  is_generating_ = false;
  return is_processed;
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
const std::string &Qwen3VLPipeline::compute_backend() const {
  return config_.compute_backend();
}
const std::string &Qwen3VLPipeline::error_message() const {
  return error_message_;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
