#include "Omni-Runtime/models/qwen3_vl/model.h"
#include "Omni-Runtime/models/qwen3_vl/executor.h"
#include "Omni-Runtime/models/qwen3_vl/weight_schema.h"

#include <memory>
#include <string>

#include "Omni-Runtime/utils/utils.h"
#include "Omni-Runtime/inference/model_loader/weight_loader.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {
bool Qwen3VLModel::Load(const std::string &config_path) {
  ClearError();
  data_ = Qwen3VLModelData{};
  auto &config = data_.config;
  OMNI_RETURN_VAL_IF(!config.Load(config_path), SetError(config.error_message()));
  OMNI_RETURN_VAL_IF(config.model_name() != "qwen3_vl" && config.model_name() != "qwen3vl",
                     SetError("unsupported model_name: " + config.model_name()));
  OMNI_RETURN_VAL_IF(config.graph_path().empty(), SetError("Qwen3-VL graph_path is missing"));
  std::string graph_path;
  OMNI_RETURN_VAL_IF(!config.ResolveRelativePath(config.graph_path(), &graph_path),
                     SetError("failed to resolve Qwen3-VL graph_path: " + config.graph_path()));
  OMNI_RETURN_VAL_IF(!graph_config_.Load(graph_path), SetError(graph_config_.error_message()));
  config.set_compute_backend(graph_config_.backend());
  std::string model_path;
  std::string mmproj_path;
  OMNI_RETURN_VAL_IF(!config.ResolveRelativePath(config.model_path(), &model_path) ||
                         !config.ResolveRelativePath(config.mmproj_path(), &mmproj_path),
                     SetError("failed to resolve Qwen3-VL model paths"));
  util::GGUFSerializer model_serializer;
  util::GGUFSerializer vision_serializer;
  OMNI_RETURN_VAL_IF(!model_serializer.Parse(model_path),
                     SetError(model_serializer.error_message()));
  OMNI_RETURN_VAL_IF(!vision_serializer.Parse(mmproj_path),
                     SetError(vision_serializer.error_message()));
  Qwen3VLWeightSchema schema;
  OMNI_RETURN_VAL_IF(!schema.Build(model_serializer, vision_serializer),
                     SetError(schema.error_message()));
  OMNI_RETURN_VAL_IF(!processor_.Init(model_serializer), SetError(processor_.error_message()));
  OMNI_RETURN_VAL_IF(config.model_backend() != ModelBackend::kGpu,
                     SetError("Qwen3-VL supports only the GPU backend"));
  data_.backend = backend::CreateComputeBackend(config.compute_backend());
  OMNI_RETURN_VAL_IF(data_.backend == nullptr || !data_.backend->IsAvailable(),
                     SetError("failed to initialize the compute backend"));
  WeightLoader loader;
  const auto load = [this, &loader](const util::GGUFSerializer &serializer,
                                    const std::vector<Qwen3VLWeightName> &weights) {
    for (const auto &weight : weights) {
      if (data_.weights.contains(weight.name)) {
        continue;
      }
      memory::TensorPtr tensor;
      OMNI_RETURN_VAL_IF(!loader.LoadTensor(serializer, weight.name, data_.config.compute_type(),
                                            data_.backend, &tensor),
                         SetError(loader.error_message()));
      data_.gpu_memory_bytes += tensor->size_bytes();
      data_.weights.emplace(weight.name, std::move(tensor));
    }
    return true;
  };
  OMNI_RETURN_VAL_IF(!load(model_serializer, schema.model_weights()) ||
                         !load(vision_serializer, schema.vision_weights()),
                     false);
  data_.model_config = schema.model_config();
  data_.vision_config = schema.vision_config();
  output_weight_name_ = schema.output_weight_name();
  OMNI_RETURN_VAL_IF(!data_.backend->InitializeCompute(),
                     SetError("failed to initialize Qwen3-VL compute backend"));
  data_.has_weights_loaded = true;
  return true;
}

bool Qwen3VLModel::CreateExecutor(std::unique_ptr<ModelExecutor> *const executor) const {
  OMNI_RETURN_VAL_IF_LOG(executor == nullptr || !data_.has_weights_loaded, false, ERROR,
                         "Qwen3-VL model is not loaded or executor output is null");
  *executor = std::make_unique<Qwen3VLExecutor>(*this);
  return true;
}

const ModelDescriptor &Qwen3VLModel::descriptor() const {
  static const ModelDescriptor descriptor{"qwen3vl", ModelTask::kTextGeneration, true};
  return descriptor;
}

const std::string &Qwen3VLModel::compute_backend() const {
  return data_.config.compute_backend();
}

const Qwen3VLConfig &Qwen3VLModel::model_config() const {
  return data_.model_config;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
