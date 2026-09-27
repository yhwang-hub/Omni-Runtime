#pragma once

#include <unordered_map>

#include "Omni-Runtime/models/model.h"
#include "Omni-Runtime/inference/common/inference_config/inference_config.h"
#include "Omni-Runtime/inference/inference_graph/operator_graph.h"
#include "Omni-Runtime/inference/model_loader/weight_define.h"
#include "Omni-Runtime/models/components/qwen_multimodal/processor.h"
#include "Omni-Runtime/models/qwen3_5/config_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_5 {

class Qwen35Model final : public Model {
public:
  bool Load(const std::string &config_path) override;
  bool CreateExecutor(std::unique_ptr<ModelExecutor> *const executor) const override;
  const ModelDescriptor &descriptor() const override;
  const std::string &compute_backend() const override;

  const Qwen35Config &model_config() const {
    return model_config_;
  }

  const InferenceConfig &config() const {
    return config_;
  }

  const OperatorGraphConfig &graph_config() const {
    return graph_config_;
  }

  const qwen3::QwenProcessor &processor() const {
    return processor_;
  }

  const std::unordered_map<std::string, WeightStorage> &weights() const {
    return weights_;
  }

  const std::shared_ptr<backend::Backend> &backend() const {
    return backend_;
  }

  size_t gpu_memory_bytes() const {
    return gpu_memory_bytes_;
  }

private:
  bool UploadWeights(const util::GGUFSerializer &serializer);

  InferenceConfig config_;
  OperatorGraphConfig graph_config_;
  qwen3::QwenProcessor processor_;
  Qwen35Config model_config_;
  std::shared_ptr<backend::Backend> backend_;
  std::unordered_map<std::string, WeightStorage> weights_;
  size_t gpu_memory_bytes_ = 0U;
  bool is_loaded_ = false;
};

} // namespace qwen3_5
} // namespace inference
} // namespace omni_runtime
