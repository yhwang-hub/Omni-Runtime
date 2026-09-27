#pragma once

#include <memory>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/inference/inference_graph/inference_graph.h"
#include "Omni-Runtime/inference/inference_graph/operator_graph_define.h"

namespace omni_runtime {
namespace inference {

class OperatorGraphConfig final {
public:
  bool Load(const std::string &path);
  bool Append(const std::string &section, const OperatorGraphBindings &bindings,
              const OperatorIterationBinder &bind_iteration,
              std::vector<backend::ComputeNode> *const nodes);
  const std::string &backend() const;
  const std::string &error_message() const;

private:
  bool ValidateSequence(const YAML::Node &sequence, const size_t depth);
  bool ExpandSequence(const YAML::Node &sequence, const OperatorGraphBindings &bindings,
                      const OperatorIterationBinder &bind_iteration, const size_t depth,
                      std::vector<uint32_t> *const dependencies,
                      std::vector<backend::ComputeNode> *const nodes);
  bool SetError(const std::string &message);

  YAML::Node document_;
  bool is_loaded_ = false;
  std::string backend_;
  std::string error_message_;
};

class OperatorGraph final {
public:
  bool Prepare(const std::string &name, const std::shared_ptr<backend::Backend> &backend,
               const backend::ScalarType type, const std::vector<backend::ComputeNode> &nodes);
  bool Run();
  const std::string &error_message() const;

private:
  InferenceGraph graph_;
  bool is_prepared_ = false;
  std::string error_message_;
};

} // namespace inference
} // namespace omni_runtime
