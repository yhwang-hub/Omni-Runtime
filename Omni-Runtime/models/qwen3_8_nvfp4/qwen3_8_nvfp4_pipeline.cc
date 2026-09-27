#include "Omni-Runtime/models/qwen3_8_nvfp4/qwen3_8_nvfp4_pipeline.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <utility>

#include "Omni-Runtime/inference/common/text_stream/text_stream.h"
#include "Omni-Runtime/inference/tokenizer/chat_template.h"
#include "Omni-Runtime/inference/tokenizer/gguf_tokenizer_loader.h"
#include "Omni-Runtime/inference/tokenizer/tokenizer_factory.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime::inference::qwen3_8_nvfp4 {

Qwen38Pipeline::Qwen38Pipeline() = default;
Qwen38Pipeline::~Qwen38Pipeline() = default;

bool Qwen38Pipeline::Init(const std::string &config_path) {
  OMNI_RETURN_VAL_IF_LOG(is_generating_, false, ERROR, "Cannot reinitialize during generation");
  is_initialized_ = false;
  error_message_.clear();
  decode_plan_.reset();
  timer_.reset();
  weights_.clear();
  buffers_.clear();
  states_.clear();
  tokenizer_.reset();
  backend_.reset();
  gpu_memory_bytes_ = 0U;
  inference_config_ = InferenceConfig{};
  model_config_ = Qwen38Config{};
  OMNI_RETURN_VAL_IF(!inference_config_.Load(config_path),
                     SetError(inference_config_.error_message()));
  OMNI_RETURN_VAL_IF(inference_config_.model_name() != "qwen3_8_nvfp4" ||
                         inference_config_.compute_type() != ModelWeightType::kFloat32 ||
                         inference_config_.max_context_tokens() <= 0 ||
                         inference_config_.max_context_tokens() > 262144,
                     SetError("qwen3_8_nvfp4 requires FP32 activations and a valid context size"));
  std::string graph_path;
  std::string model_path;
  std::string vision_path;
  OMNI_RETURN_VAL_IF(
      !inference_config_.ResolveRelativePath(inference_config_.graph_path(), &graph_path) ||
          !inference_config_.ResolveRelativePath(inference_config_.model_path(), &model_path) ||
          !inference_config_.ResolveRelativePath(inference_config_.mmproj_path(), &vision_path),
      SetError("cannot resolve model, mmproj or operator graph path"));
  OMNI_RETURN_VAL_IF(!graph_config_.Load(graph_path), SetError(graph_config_.error_message()));
  std::string backend_name = graph_config_.backend();
  const char *const override_backend = std::getenv("OMNI_BACKEND");
  if (override_backend != nullptr && override_backend[0] != '\0' &&
      std::string(override_backend) != "auto") {
    backend_name = override_backend;
  }
  OMNI_RETURN_VAL_IF(backend_name != "cuda" && backend_name != "opencl",
                     SetError("unsupported compute backend: " + backend_name));
  inference_config_.set_compute_backend(backend_name);
  backend_ = backend::CreateComputeBackend(backend_name);
  OMNI_RETURN_VAL_IF(backend_ == nullptr || !backend_->InitializeCompute(),
                     SetError("compute backend is unavailable: " + backend_name));
  util::GGUFSerializer model;
  util::GGUFSerializer vision;
  OMNI_RETURN_VAL_IF(!model.Parse(model_path), SetError(model.error_message()));
  OMNI_RETURN_VAL_IF(!vision.Parse(vision_path), SetError(vision.error_message()));
  OMNI_RETURN_VAL_IF(!LoadMetadata(model, vision), false);
  TokenizerModel tokenizer_model;
  GGUFTokenizerLoader loader;
  OMNI_RETURN_VAL_IF(!loader.Load(model, &tokenizer_model), SetError(loader.error_message()));
  OMNI_RETURN_VAL_IF(tokenizer_model.vocabulary.size() >
                         static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                     SetError("tokenizer vocabulary exceeds int32 range"));
  model_config_.vocabulary_size = static_cast<int32_t>(tokenizer_model.vocabulary.size());
  const std::pair<std::string, int32_t *> special[] = {
      {"<|im_end|>", &model_config_.end_of_sequence_id},
      {"<|endoftext|>", &model_config_.end_of_text_id},
      {"<|image_pad|>", &model_config_.image_id},
  };
  for (const auto &[name, id] : special) {
    const auto token = tokenizer_model.special_tokens.find(name);
    OMNI_RETURN_VAL_IF(token == tokenizer_model.special_tokens.end(),
                       SetError("missing tokenizer special token: " + name));
    *id = token->second;
  }
  tokenizer_ = TokenizerFactory::Create(tokenizer_model.type);
  OMNI_RETURN_VAL_IF(tokenizer_ == nullptr || !tokenizer_->Init(tokenizer_model),
                     SetError("cannot initialize GGUF tokenizer"));
  OMNI_RETURN_VAL_IF(!LoadWeights(model) || !LoadWeights(vision), false);
  OMNI_RETURN_VAL_IF(!ValidateWeights(), SetError("incompatible GGUF tensor registry"));
  timer_ = std::make_unique<backend::EventTimer>();
  OMNI_RETURN_VAL_IF(!timer_->Initialize(backend_) || !AllocateStorage() || !ResetState() ||
                         !PrepareDecode(),
                     SetError(error_message_.empty() ? "cannot initialize model execution storage"
                                                     : error_message_));
  is_initialized_ = true;
  return true;
}

bool Qwen38Pipeline::Allocate(const std::string &name, const size_t bytes) {
  OMNI_RETURN_VAL_IF(buffers_.contains(name) || bytes == 0U,
                     SetError("invalid workspace allocation: " + name));
  auto buffer = memory::Buffer::CreateOwnedBuffer(backend_, bytes, backend::MemoryKind::kDevice);
  OMNI_RETURN_VAL_IF(buffer == nullptr, SetError("failed to allocate workspace: " + name));
  gpu_memory_bytes_ += bytes;
  buffers_.emplace(name, std::move(buffer));
  return true;
}

bool Qwen38Pipeline::AllocateStorage() {
  const auto &c = model_config_;
  const auto &v = c.vision;
  const size_t batch = kPrefillBatchTokens;
  const size_t capacity = inference_config_.max_context_tokens();
  const size_t query = static_cast<size_t>(c.query_head_count) * c.head_size;
  const size_t kv = static_cast<size_t>(c.key_value_head_count) * c.head_size;
  const size_t recurrent = static_cast<size_t>(c.recurrent_value_heads) * c.recurrent_head_size;
  const size_t channels =
      recurrent + 2U * static_cast<size_t>(c.recurrent_key_heads) * c.recurrent_head_size;
  const size_t patches = capacity * v.merge_size * v.merge_size;
  const size_t vision_hidden = patches * v.hidden_size;
  const std::pair<std::string, size_t> workspaces[] = {
      {"tokens", batch * sizeof(int32_t)},
      {"positions", batch * 3U * sizeof(int32_t)},
      {"cache_position", sizeof(int32_t)},
      {"sample", sizeof(int32_t)},
      {"hidden", batch * c.hidden_size * sizeof(float)},
      {"residual", batch * c.hidden_size * sizeof(float)},
      {"normalized", batch * c.hidden_size * sizeof(float)},
      {"normalized_q8", batch * c.hidden_size * sizeof(float)},
      {"normalized_q8_scales", batch * (c.hidden_size / 32) * sizeof(float)},
      {"normalized_ffn", batch * c.hidden_size * sizeof(float)},
      {"normalized_ffn_scales", batch * (c.hidden_size / 16) * sizeof(float)},
      {"projected", batch * c.hidden_size * sizeof(float)},
      {"qg", batch * query * 2U * sizeof(float)},
      {"query", batch * query * sizeof(float)},
      {"query_normed", batch * query * sizeof(float)},
      {"attention_gate", batch * query * sizeof(float)},
      {"key", batch * kv * sizeof(float)},
      {"key_normed", batch * kv * sizeof(float)},
      {"value", batch * kv * sizeof(float)},
      {"attention", batch * query * sizeof(float)},
      {"attention_quantized", batch * query * sizeof(float)},
      {"attention_scales", batch * (query / 32U) * sizeof(float)},
      {"gate", batch * c.feed_forward_size * sizeof(float)},
      {"up", batch * c.feed_forward_size * sizeof(float)},
      {"ffn_quantized", batch * c.feed_forward_size * sizeof(float)},
      {"ffn_scales", batch * (c.feed_forward_size / 16) * sizeof(float)},
      {"qkv", batch * channels * sizeof(float)},
      {"convolved", batch * channels * sizeof(float)},
      {"z", batch * recurrent * sizeof(float)},
      {"alpha", batch * c.recurrent_value_heads * sizeof(float)},
      {"beta", batch * c.recurrent_value_heads * sizeof(float)},
      {"delta", batch * recurrent * sizeof(float)},
      {"delta_normed", batch * recurrent * sizeof(float)},
      {"delta_quantized", batch * recurrent * sizeof(float)},
      {"delta_scales", batch * (recurrent / 32U) * sizeof(float)},
      {"output_quantized", static_cast<size_t>(c.hidden_size) * sizeof(float)},
      {"output_scales", static_cast<size_t>(c.hidden_size / 32) * sizeof(float)},
      {"logits", static_cast<size_t>(c.vocabulary_size) * sizeof(float)},
      {"decode_partials",
       ((capacity + 127U) / 128U) * c.query_head_count * (c.head_size + 2U) * sizeof(float)},
      {"vision_source", patches * v.patch_value_count()},
      {"vision_patches", patches * v.patch_value_count() * sizeof(float)},
      {"vision_patch_embedding", vision_hidden * sizeof(float)},
      {"vision_hidden", vision_hidden * sizeof(float)},
      {"vision_residual", vision_hidden * sizeof(float)},
      {"vision_normalized", vision_hidden * sizeof(float)},
      {"vision_qkv", vision_hidden * 3U * sizeof(float)},
      {"vision_query", vision_hidden * sizeof(float)},
      {"vision_key", vision_hidden * sizeof(uint16_t)},
      {"vision_value", vision_hidden * sizeof(uint16_t)},
      {"vision_attention", vision_hidden * sizeof(float)},
      {"vision_projected", vision_hidden * sizeof(float)},
      {"vision_feed_forward",
       patches * std::max(v.feed_forward_size, v.merged_hidden_size()) * sizeof(float)},
      {"image_embedding", capacity * c.hidden_size * sizeof(float)},
  };
  for (const auto &[name, bytes] : workspaces) {
    OMNI_RETURN_VAL_IF(!Allocate(name, bytes), false);
  }
  states_.resize(c.layer_count);
  for (int32_t layer = 0; layer < c.layer_count; ++layer) {
    const auto allocate = [this](const size_t bytes, memory::BufferPtr *const buffer) {
      OMNI_RETURN_VAL_IF(buffer == nullptr, false);
      *buffer = memory::Buffer::CreateOwnedBuffer(backend_, bytes, backend::MemoryKind::kDevice);
      OMNI_RETURN_VAL_IF(*buffer == nullptr, false);
      gpu_memory_bytes_ += bytes;
      return true;
    };
    auto &state = states_[layer];
    if (c.IsRecurrent(layer)) {
      OMNI_RETURN_VAL_IF(
          !allocate(channels * (c.convolution_size - 1) * sizeof(float), &state.convolution) ||
              !allocate(recurrent * c.recurrent_head_size * sizeof(float), &state.recurrent),
          SetError("failed to allocate recurrent layer state"));
    } else {
      OMNI_RETURN_VAL_IF(!allocate(capacity * kv * sizeof(uint16_t), &state.key) ||
                             !allocate(capacity * kv * sizeof(uint16_t), &state.value),
                         SetError("failed to allocate attention KV cache"));
    }
  }
  return true;
}

bool Qwen38Pipeline::ResetState() {
  for (auto &state : states_) {
    if (state.recurrent != nullptr) {
      OMNI_RETURN_VAL_IF(!state.recurrent->Fill(0U) || !state.convolution->Fill(0U),
                         SetError("failed to reset recurrent state"));
    }
  }
  return buffers_.at("cache_position")->Fill(0U);
}

bool Qwen38Pipeline::Generate(const GenerationRequest &request, GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(result == nullptr, SetError("generation output is null"));
  OMNI_RETURN_VAL_IF_LOG(is_generating_, false, ERROR, "Cannot reenter generation");
  OMNI_RETURN_VAL_IF(!is_initialized_, SetError("pipeline is not initialized"));
  OMNI_RETURN_VAL_IF(request.prompt.empty() || request.max_new_tokens <= 0 ||
                         request.max_new_tokens >= inference_config_.max_context_tokens(),
                     SetError("invalid prompt or maximum generated token count"));
  OMNI_RETURN_VAL_IF(!request.image_path.empty() &&
                         !std::filesystem::is_regular_file(request.image_path),
                     SetError("input image does not exist"));
  error_message_.clear();
  *result = GenerationResult{};
  OMNI_RETURN_VAL_IF(!ResetState(), false);
  Prompt prompt;
  int32_t first_token = -1;
  is_generating_ = true;
  const bool is_successful =
      EncodeImage(request, &prompt, result) && PreparePrompt(request, &prompt) &&
      Prefill(prompt, &first_token, result) && Decode(request, first_token, result);
  is_generating_ = false;
  return is_successful;
}

bool Qwen38Pipeline::PreparePrompt(const GenerationRequest &request, Prompt *const prompt) {
  OMNI_RETURN_VAL_IF(prompt == nullptr, SetError("prompt output is null"));
  std::string text;
  const Qwen3VLChatTemplate chat;
  OMNI_RETURN_VAL_IF(!chat.BuildUserPrompt(request.prompt, !request.image_path.empty(), &text) ||
                         !tokenizer_->Encode(text, &prompt->tokens),
                     SetError("prompt tokenization failed: " + tokenizer_->error_message()));
  if (prompt->image_tokens > 0) {
    const auto image =
        std::find(prompt->tokens.begin(), prompt->tokens.end(), model_config_.image_id);
    OMNI_RETURN_VAL_IF(image == prompt->tokens.end(), SetError("image placeholder is missing"));
    prompt->image_offset = static_cast<int32_t>(std::distance(prompt->tokens.begin(), image));
    prompt->tokens.insert(image, prompt->image_tokens - 1, model_config_.image_id);
  }
  OMNI_RETURN_VAL_IF(prompt->tokens.empty() ||
                         prompt->tokens.size() + static_cast<size_t>(request.max_new_tokens) >
                             static_cast<size_t>(inference_config_.max_context_tokens()),
                     SetError("prompt and generation exceed context capacity"));
  OMNI_RETURN_VAL_IF(std::any_of(prompt->tokens.begin(), prompt->tokens.end(),
                                 [this](const int32_t id) {
                                   return id < 0 || id >= model_config_.vocabulary_size;
                                 }),
                     SetError("token ID is outside the vocabulary"));
  const int32_t count = static_cast<int32_t>(prompt->tokens.size());
  prompt->positions.resize(static_cast<size_t>(count) * 3U);
  int32_t position = 0;
  for (int32_t token = 0; token < count; ++token) {
    if (prompt->image_offset >= 0 && token >= prompt->image_offset &&
        token < prompt->image_offset + prompt->image_tokens) {
      const int32_t image = token - prompt->image_offset;
      prompt->positions[token] = prompt->image_offset;
      prompt->positions[count + token] = prompt->image_offset + image / prompt->image_width;
      prompt->positions[2 * count + token] = prompt->image_offset + image % prompt->image_width;
      position = prompt->image_offset + std::max(prompt->image_height, prompt->image_width);
    } else {
      for (int32_t axis = 0; axis < 3; ++axis) {
        prompt->positions[static_cast<size_t>(axis) * count + token] = position;
      }
      ++position;
    }
  }
  prompt->next_position = position;
  return true;
}

bool Qwen38Pipeline::Decode(const GenerationRequest &request, const int32_t first_token,
                            GenerationResult *const result) {
  OMNI_RETURN_VAL_IF(result == nullptr, SetError("decode output is null"));
  TextStream stream(request.stream_callback);
  int32_t token = first_token;
  for (int32_t index = 0; index < request.max_new_tokens; ++index) {
    OMNI_RETURN_VAL_IF(token < 0 || token >= model_config_.vocabulary_size,
                       SetError("sampled token ID is outside the vocabulary"));
    result->generated_token_ids.push_back(token);
    if (token == model_config_.end_of_sequence_id || token == model_config_.end_of_text_id) {
      break;
    }
    std::string bytes;
    OMNI_RETURN_VAL_IF(!tokenizer_->Decode({token}, &bytes), SetError(tokenizer_->error_message()));
    OMNI_RETURN_VAL_IF(!stream.Append(bytes), SetError(stream.error_message()));
    if (stream.is_cancelled() || index + 1 == request.max_new_tokens) {
      break;
    }
    double elapsed_ms = 0.0;
    OMNI_RETURN_VAL_IF(!decode_plan_->Run(sizeof(token), &token, &elapsed_ms),
                       SetError("resident decode graph execution failed"));
    result->decode_time_ms += elapsed_ms;
    ++result->decode_token_count;
  }
  OMNI_RETURN_VAL_IF(!stream.Finish(), SetError(stream.error_message()));
  result->text = stream.text();
  result->is_cancelled = stream.is_cancelled();
  return true;
}

bool Qwen38Pipeline::SetError(const std::string &message) {
  error_message_ = message;
  return false;
}

const Qwen38Config &Qwen38Pipeline::model_config() const {
  return model_config_;
}

const std::string &Qwen38Pipeline::compute_backend() const {
  return inference_config_.compute_backend();
}

const std::string &Qwen38Pipeline::error_message() const {
  return error_message_;
}

} // namespace omni_runtime::inference::qwen3_8_nvfp4
