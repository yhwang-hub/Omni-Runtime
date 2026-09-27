#pragma once

#include "Omni-Runtime/models/model.h"
#include "Omni-Runtime/models/components/qwen_multimodal/processor.h"
#include "Omni-Runtime/models/qwen3_vl/model_define.h"
#include "Omni-Runtime/inference/inference_graph/operator_graph.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

class Qwen3VLModel final : public Model {
public:
  bool Load(const std::string &config_path) override;
  bool CreateExecutor(std::unique_ptr<ModelExecutor> *const executor) const override;
  const ModelDescriptor &descriptor() const override;
  const std::string &compute_backend() const override;
  const Qwen3VLConfig &model_config() const;

  const Qwen3VLModelData &data() const {
    return data_;
  }

  const OperatorGraphConfig &graph_config() const {
    return graph_config_;
  }

  const qwen3::QwenProcessor &processor() const {
    return processor_;
  }

  const std::string &output_weight_name() const {
    return output_weight_name_;
  }

private:
  OperatorGraphConfig graph_config_;
  qwen3::QwenProcessor processor_;
  std::string output_weight_name_;
  Qwen3VLModelData data_;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
