#include "Omni-Runtime/models/qwen3_5/model.h"

#include <cstdlib>

#include "Omni-Runtime/inference/tokenizer/gguf_tokenizer_loader.h"
#include "Omni-Runtime/inference/model_loader/weight_loader.h"
#include "Omni-Runtime/models/qwen3_5/executor.h"
#include "Omni-Runtime/models/qwen3_5/weight_schema.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_5 {

bool Qwen35Model::Load(const std::string &config_path) {
  ClearError();
  is_loaded_ = false;
  weights_.clear();
  backend_.reset();
  gpu_memory_bytes_ = 0U;
  model_config_ = Qwen35Config{};
  OMNI_RETURN_VAL_IF(!config_.Load(config_path), SetError(config_.error_message()));
  OMNI_RETURN_VAL_IF(
      (config_.model_name() != "qwen3_8_nvfp4" && config_.model_name() != "qwen3_8_fp8" &&
       config_.model_name() != "qwen3_5" && config_.model_name() != "qwen35") ||
          (config_.compute_type() != ModelWeightType::kFloat32 &&
           !(config_.model_name() == "qwen3_8_fp8" &&
             config_.compute_type() == ModelWeightType::kBFloat16)) ||
          config_.max_context_tokens() <= 0 || config_.max_context_tokens() > 262144,
      SetError("qwen35 requires FP32 or FP8/BF16 activations and a valid context size"));
  std::string graph_path;
  std::string model_path;
  std::string vision_path;
  OMNI_RETURN_VAL_IF(!config_.ResolveRelativePath(config_.graph_path(), &graph_path) ||
                         !config_.ResolveRelativePath(config_.model_path(), &model_path) ||
                         !config_.ResolveRelativePath(config_.mmproj_path(), &vision_path),
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
  config_.set_compute_backend(backend_name);
  backend_ = backend::CreateComputeBackend(backend_name);
  OMNI_RETURN_VAL_IF(backend_ == nullptr || !backend_->InitializeCompute(),
                     SetError("compute backend is unavailable: " + backend_name));
  util::GGUFSerializer model;
  util::GGUFSerializer vision;
  OMNI_RETURN_VAL_IF(!model.Parse(model_path), SetError(model.error_message()));
  OMNI_RETURN_VAL_IF(!vision.Parse(vision_path), SetError(vision.error_message()));
  TokenizerModel tokenizer;
  GGUFTokenizerLoader tokenizer_loader;
  OMNI_RETURN_VAL_IF(!tokenizer_loader.Load(model, &tokenizer),
                     SetError(tokenizer_loader.error_message()));
  Qwen35WeightSchema schema;
  OMNI_RETURN_VAL_IF(!schema.Build(model, vision, tokenizer), SetError(schema.error_message()));
  model_config_ = schema.model_config();
  weights_ = schema.weights();
  OMNI_RETURN_VAL_IF(!processor_.Init(tokenizer), SetError(processor_.error_message()));
  OMNI_RETURN_VAL_IF(!UploadWeights(model) || !UploadWeights(vision), false);
  is_loaded_ = true;
  return true;
}

bool Qwen35Model::CreateExecutor(std::unique_ptr<ModelExecutor> *const executor) const {
  OMNI_RETURN_VAL_IF_LOG(executor == nullptr || !is_loaded_, false, ERROR,
                         "Qwen3.5 model is not loaded or executor output is null");
  *executor = std::make_unique<Qwen35Executor>(*this);
  return true;
}

bool Qwen35Model::UploadWeights(const util::GGUFSerializer &serializer) {
  WeightLoader loader;
  for (const auto &tensor : serializer.model().tensors) {
    if (tensor.name.ends_with(".input_scale") || tensor.name.ends_with(".scale")) {
      continue;
    }
    const auto found = weights_.find(tensor.name);
    OMNI_RETURN_VAL_IF(found == weights_.end(),
                       SetError("unregistered GGUF tensor: " + tensor.name));
    const bool is_float_weight =
        tensor.name.ends_with(".weight_scale_inv") || tensor.name.ends_with("ssm_conv1d.weight") ||
        tensor.name.ends_with("ssm_a") || tensor.name.ends_with("ssm_a_log") ||
        tensor.name.ends_with("ssm_dt.bias") ||
        (tensor.name.ends_with("norm.weight") && !tensor.name.ends_with("ssm_norm.weight"));
    if (config_.compute_type() == ModelWeightType::kBFloat16 &&
        found->second.type != util::GGUFTensorType::FP8_E4M3 && !is_float_weight) {
      memory::TensorPtr converted;
      OMNI_RETURN_VAL_IF(!loader.LoadTensor(serializer, tensor.name, ModelWeightType::kBFloat16,
                                            backend_, &converted),
                         SetError(loader.error_message()));
      found->second.storage = converted->buffer();
      found->second.type = util::GGUFTensorType::BFLOAT16;
    } else {
      OMNI_RETURN_VAL_IF(
          !loader.LoadEncoded(serializer, tensor.name, backend_, &found->second.storage),
          SetError(loader.error_message()));
    }
    gpu_memory_bytes_ += found->second.storage->size_bytes();
  }
  return true;
}

const ModelDescriptor &Qwen35Model::descriptor() const {
  static const ModelDescriptor descriptor{"qwen35", ModelTask::kTextGeneration, true};
  return descriptor;
}

const std::string &Qwen35Model::compute_backend() const {
  return config_.compute_backend();
}

} // namespace qwen3_5
} // namespace inference
} // namespace omni_runtime
