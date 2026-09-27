#include "Omni-Runtime/inference/inference_graph/operator_graph.h"

#include <cmath>
#include <filesystem>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace {

const std::unordered_map<std::string, backend::ComputeOperation> &Operations() {
  static const std::unordered_map<std::string, backend::ComputeOperation> operations = {
      {"Embedding", backend::Embedding{}},
      {"Linear", backend::Linear{}},
      {"QuantizedLinear", backend::QuantizedLinear{}},
      {"QuantizeActivation", backend::QuantizeActivation{}},
      {"QuantizedEmbedding", backend::QuantizedEmbedding{}},
      {"CausalConv1d", backend::CausalConv1d{}},
      {"GatedDeltaNet", backend::GatedDeltaNet{}},
      {"SigmoidGate", backend::SigmoidGate{}},
      {"PartialMultiAxisRope", backend::PartialMultiAxisRope{}},
      {"RmsNorm", backend::RmsNorm{}},
      {"LayerNorm", backend::LayerNorm{}},
      {"MultiAxisRope", backend::MultiAxisRope{}},
      {"StoreKvCache", backend::StoreKvCache{}},
      {"Attention", backend::Attention{}},
      {"ResidualRmsNorm", backend::ResidualRmsNorm{}},
      {"NormalizeRopeCache", backend::NormalizeRopeCache{}},
      {"Add", backend::Add{}},
      {"SwiGLU", backend::SwiGLU{}},
      {"Argmax", backend::Argmax{}},
      {"AdvanceSequence", backend::AdvanceSequence{}},
      {"ImagePatchPreprocess", backend::ImagePatchPreprocess{}},
      {"PairedPatchEmbedding", backend::PairedPatchEmbedding{}},
      {"GridPositionEmbedding", backend::GridPositionEmbedding{}},
      {"Bias", backend::Bias{}},
      {"BiasResidual", backend::BiasResidual{}},
      {"GeluBias", backend::GeluBias{}},
      {"GridQkvRope", backend::GridQkvRope{}},
      {"AdaRmsNorm", backend::AdaRmsNorm{}},
      {"FlowStep", backend::FlowStep{}},
      {"GatedAdd", backend::GatedAdd{}},
      {"SplitGatedQuery", backend::SplitGatedQuery{}},
      {"FusedFfn", backend::FusedFfn{}},
      {"MultiHeadAttention", backend::MultiHeadAttention{}},
      {"PatchEmbedding", backend::PatchEmbedding{}},
      {"PatchMerge", backend::PatchMerge{}}};
  return operations;
}

template <typename Operation, typename Visitor>
bool VisitFields(Operation *const operation, const Visitor &visitor) {
  OMNI_RETURN_VAL_IF_LOG(operation == nullptr, false, ERROR, "Operator fields are null");
#define OMNI_VISIT_FIELD(field)                                                                    \
  if constexpr (requires { operation->field; }) {                                                  \
    OMNI_RETURN_VAL_IF(!visitor(#field, &operation->field), false);                                \
  }
  OMNI_VISIT_FIELD(input)
  OMNI_VISIT_FIELD(weight)
  OMNI_VISIT_FIELD(bias)
  OMNI_VISIT_FIELD(output)
  OMNI_VISIT_FIELD(rows)
  OMNI_VISIT_FIELD(inner)
  OMNI_VISIT_FIELD(columns)
  OMNI_VISIT_FIELD(hidden_size)
  OMNI_VISIT_FIELD(epsilon)
  OMNI_VISIT_FIELD(indices)
  OMNI_VISIT_FIELD(token_count)
  OMNI_VISIT_FIELD(query)
  OMNI_VISIT_FIELD(key)
  OMNI_VISIT_FIELD(value)
  OMNI_VISIT_FIELD(position_ids)
  OMNI_VISIT_FIELD(tokens)
  OMNI_VISIT_FIELD(query_heads)
  OMNI_VISIT_FIELD(key_value_heads)
  OMNI_VISIT_FIELD(head_dim)
  OMNI_VISIT_FIELD(temporal_section)
  OMNI_VISIT_FIELD(height_section)
  OMNI_VISIT_FIELD(width_section)
  OMNI_VISIT_FIELD(frequency_base)
  OMNI_VISIT_FIELD(query_output)
  OMNI_VISIT_FIELD(key_output)
  OMNI_VISIT_FIELD(key_value_size)
  OMNI_VISIT_FIELD(start_position)
  OMNI_VISIT_FIELD(cache_capacity)
  OMNI_VISIT_FIELD(key_cache)
  OMNI_VISIT_FIELD(value_cache)
  OMNI_VISIT_FIELD(cache_position)
  OMNI_VISIT_FIELD(query_token_count)
  OMNI_VISIT_FIELD(query_head_count)
  OMNI_VISIT_FIELD(key_value_head_count)
  OMNI_VISIT_FIELD(cached_token_count)
  OMNI_VISIT_FIELD(is_causal)
  OMNI_VISIT_FIELD(decode_partials)
  OMNI_VISIT_FIELD(positions)
  OMNI_VISIT_FIELD(residual)
  OMNI_VISIT_FIELD(residual_output)
  OMNI_VISIT_FIELD(query_weight)
  OMNI_VISIT_FIELD(key_weight)
  OMNI_VISIT_FIELD(lhs)
  OMNI_VISIT_FIELD(rhs)
  OMNI_VISIT_FIELD(element_count)
  OMNI_VISIT_FIELD(gate)
  OMNI_VISIT_FIELD(up)
  OMNI_VISIT_FIELD(logits)
  OMNI_VISIT_FIELD(pixel_count)
  OMNI_VISIT_FIELD(source)
  OMNI_VISIT_FIELD(source_width_pixels)
  OMNI_VISIT_FIELD(source_height_pixels)
  OMNI_VISIT_FIELD(patch_size)
  OMNI_VISIT_FIELD(destination)
  OMNI_VISIT_FIELD(input_width_pixels)
  OMNI_VISIT_FIELD(input_height_pixels)
  OMNI_VISIT_FIELD(is_letterbox)
  OMNI_VISIT_FIELD(first_weight)
  OMNI_VISIT_FIELD(second_weight)
  OMNI_VISIT_FIELD(height)
  OMNI_VISIT_FIELD(width)
  OMNI_VISIT_FIELD(channels)
  OMNI_VISIT_FIELD(output_channels)
  OMNI_VISIT_FIELD(spatial_patch)
  OMNI_VISIT_FIELD(position_weight)
  OMNI_VISIT_FIELD(grid_height)
  OMNI_VISIT_FIELD(grid_width)
  OMNI_VISIT_FIELD(position_grid_size)
  OMNI_VISIT_FIELD(merge_size)
  OMNI_VISIT_FIELD(row_size)
  OMNI_VISIT_FIELD(is_approximate)
  OMNI_VISIT_FIELD(qkv)
  OMNI_VISIT_FIELD(head_count)
  OMNI_VISIT_FIELD(head_size)
  OMNI_VISIT_FIELD(scale)
  OMNI_VISIT_FIELD(shift)
  OMNI_VISIT_FIELD(state)
  OMNI_VISIT_FIELD(velocity)
  OMNI_VISIT_FIELD(dt)
  OMNI_VISIT_FIELD(query_gate)
  OMNI_VISIT_FIELD(fc1_weight)
  OMNI_VISIT_FIELD(fc1_bias)
  OMNI_VISIT_FIELD(fc2_weight)
  OMNI_VISIT_FIELD(fc2_bias)
  OMNI_VISIT_FIELD(input_size)
  OMNI_VISIT_FIELD(intermediate_size)
  OMNI_VISIT_FIELD(output_size)
  OMNI_VISIT_FIELD(heads)
  OMNI_VISIT_FIELD(frames)
  OMNI_VISIT_FIELD(temporal_patch)
  OMNI_VISIT_FIELD(weight_type)
  OMNI_VISIT_FIELD(weight_scale)
  OMNI_VISIT_FIELD(vocabulary_size)
  OMNI_VISIT_FIELD(kernel_size)
  OMNI_VISIT_FIELD(alpha)
  OMNI_VISIT_FIELD(beta)
  OMNI_VISIT_FIELD(decay_weight)
  OMNI_VISIT_FIELD(time_bias)
  OMNI_VISIT_FIELD(key_head_count)
  OMNI_VISIT_FIELD(value_head_count)
  OMNI_VISIT_FIELD(rotary_dim)
  OMNI_VISIT_FIELD(activation_scales)
  OMNI_VISIT_FIELD(activation_block_size)
  OMNI_VISIT_FIELD(is_decode)
  OMNI_VISIT_FIELD(scales)
  OMNI_VISIT_FIELD(is_fp16_cache)
  OMNI_VISIT_FIELD(is_fp16_attention)
  OMNI_VISIT_FIELD(is_ggml_softmax)
  OMNI_VISIT_FIELD(is_fp16_accumulation)
  OMNI_VISIT_FIELD(is_fp16_key_value)
  OMNI_VISIT_FIELD(is_ggml_interpolation)
  OMNI_VISIT_FIELD(is_ggml_normalization)
  OMNI_VISIT_FIELD(is_ggml_rope)
  OMNI_VISIT_FIELD(is_ggml_gelu)
  OMNI_VISIT_FIELD(is_ggml_rounding)
  OMNI_VISIT_FIELD(is_ggml_silu)
  OMNI_VISIT_FIELD(is_packed_nvfp4)
  OMNI_VISIT_FIELD(is_packed_q8)
  OMNI_VISIT_FIELD(is_ggml_split_k)
  OMNI_VISIT_FIELD(is_fp16_projection)
#undef OMNI_VISIT_FIELD
  return true;
}

bool ExpandName(const std::string &source, const OperatorGraphBindings &bindings,
                std::string *const name) {
  OMNI_RETURN_VAL_IF(name == nullptr, false);
  *name = source;
  size_t position = 0U;
  while ((position = name->find('{', position)) != std::string::npos) {
    const size_t end = name->find('}', position);
    OMNI_RETURN_VAL_IF_LOG(end == std::string::npos, false, ERROR,
                           "Unclosed graph variable: " << source);
    const std::string variable = name->substr(position + 1U, end - position - 1U);
    const auto iterator = bindings.values.find(variable);
    OMNI_RETURN_VAL_IF_LOG(iterator == bindings.values.end() ||
                               !std::holds_alternative<int32_t>(iterator->second),
                           false, ERROR, "Unknown graph index: " << variable);
    const std::string value = std::to_string(std::get<int32_t>(iterator->second));
    name->replace(position, end - position + 1U, value);
    position += value.size();
  }
  return true;
}

template <typename T>
bool BindValue(const YAML::Node &node, const OperatorGraphBindings &bindings, T *const value) {
  OMNI_RETURN_VAL_IF(value == nullptr, false);
  std::string name;
  OMNI_RETURN_VAL_IF(!node.IsScalar() || !ExpandName(node.as<std::string>(), bindings, &name),
                     false);
  const auto iterator = bindings.values.find(name);
  if (iterator != bindings.values.end()) {
    const T *const bound = std::get_if<T>(&iterator->second);
    OMNI_RETURN_VAL_IF_LOG(bound == nullptr, false, ERROR, "Graph binding type mismatch: " << name);
    *value = *bound;
    return true;
  }
  if constexpr (std::is_same_v<T, backend::DeviceView>) {
    OMNI_RETURN_VAL_IF_LOG(!bindings.resolve_view || !bindings.resolve_view(name, value) ||
                               !value->IsValid(),
                           false, ERROR, "Unknown graph tensor: " << name);
  } else {
    if constexpr (std::is_unsigned_v<T>) {
      OMNI_RETURN_VAL_IF_LOG(name.starts_with('-'), false, ERROR,
                             "Negative unsigned graph scalar: " << name);
    }
    *value = node.as<T>();
    if constexpr (std::is_floating_point_v<T>) {
      OMNI_RETURN_VAL_IF_LOG(!std::isfinite(*value), false, ERROR,
                             "Nonfinite graph scalar: " << name);
    }
  }
  return true;
}

bool ValidateKeys(const YAML::Node &node, const std::set<std::string> &allowed) {
  OMNI_RETURN_VAL_IF_LOG(!node.IsMap(), false, ERROR, "Graph entry must be a map");
  std::set<std::string> seen;
  for (const auto &entry : node) {
    const std::string key = entry.first.as<std::string>();
    OMNI_RETURN_VAL_IF_LOG(!allowed.contains(key) || !seen.insert(key).second, false, ERROR,
                           "Unknown or duplicate graph field: " << key);
  }
  return true;
}

bool IsOptionalField(const std::string &name) {
  return name == "cache_position" || name == "decode_partials" || name == "activation_scales" ||
         name == "activation_block_size" || name == "is_fp16_cache" ||
         name == "is_fp16_attention" || name == "is_ggml_softmax" ||
         name == "is_fp16_accumulation" || name == "is_fp16_key_value" ||
         name == "is_fp16_projection" || name == "is_letterbox" ||
         name == "is_ggml_interpolation" || name == "is_ggml_normalization" ||
         name == "is_ggml_rope" || name == "is_ggml_gelu" || name == "is_ggml_rounding" ||
         name == "is_ggml_silu" || name == "is_packed_nvfp4" || name == "is_packed_q8" ||
         name == "is_ggml_split_k";
}

class ComputeOperatorNode final : public graph::Node {
public:
  explicit ComputeOperatorNode(const std::shared_ptr<backend::Backend> &backend,
                               const backend::ScalarType type,
                               const backend::ComputeOperation &operation)
      : backend_(backend), type_(type), operation_(operation) {
  }

protected:
  graph::Status Run() final {
    OMNI_RETURN_VAL_IF_LOG(backend_ == nullptr || !backend_->Compute(type_, operation_),
                           graph::InternalStatus("Operator failed: " + name()), ERROR,
                           "Operator failed: " << name());
    return graph::Status();
  }

private:
  std::shared_ptr<backend::Backend> backend_;
  backend::ScalarType type_;
  backend::ComputeOperation operation_;
};

} // namespace

bool OperatorGraphConfig::Load(const std::string &path) {
  error_message_.clear();
  is_loaded_ = false;
  backend_.clear();
  document_.reset();
  try {
    const YAML::Node root = YAML::LoadFile(path);
    OMNI_RETURN_VAL_IF(!root.IsMap(), SetError("invalid graph configuration: " + path));
    backend_ = root["backend"].as<std::string>();
    OMNI_RETURN_VAL_IF(backend_ != "cuda" && backend_ != "opencl",
                       SetError("graph backend must be cuda or opencl"));
    if (root["graph"]) {
      OMNI_RETURN_VAL_IF(root.size() != 2U,
                         SetError("graph selector only accepts backend and graph"));
      const auto graph_path =
          std::filesystem::path(path).parent_path() / root["graph"].as<std::string>();
      document_ = YAML::LoadFile(graph_path.string());
      OMNI_RETURN_VAL_IF(!document_.IsMap() || document_["graph"],
                         SetError("graph selector must reference a graph definition"));
    } else {
      document_ = root;
    }
    std::set<std::string> fields;
    for (const auto &entry : document_) {
      const std::string section = entry.first.as<std::string>();
      OMNI_RETURN_VAL_IF(!fields.insert(section).second,
                         SetError("duplicate graph section: " + section));
      if (section == "backend") {
        continue;
      }
      OMNI_RETURN_VAL_IF(!ValidateSequence(entry.second, 0U),
                         SetError("invalid operator graph section: " + section));
    }
    OMNI_RETURN_VAL_IF(fields.size() < 2U, SetError("operator graph has no sections"));
    if (root["graph"]) {
      OMNI_RETURN_VAL_IF(!ValidateKeys(root, {"backend", "graph"}),
                         SetError("invalid graph selector"));
    }
  } catch (const YAML::Exception &error) {
    return SetError("graph YAML " + path + ": " + error.what());
  }
  is_loaded_ = true;
  return true;
}

bool OperatorGraphConfig::ValidateSequence(const YAML::Node &sequence, const size_t depth) {
  OMNI_RETURN_VAL_IF_LOG(depth > 16U || !sequence.IsSequence() || sequence.size() == 0U, false,
                         ERROR, "Graph needs a nonempty sequence (maximum nesting: 16)");
  for (const auto &node : sequence) {
    if (node.IsSequence()) {
      OMNI_RETURN_VAL_IF(!ValidateSequence(node, depth + 1U), false);
      continue;
    }
    OMNI_RETURN_VAL_IF(!node.IsMap(), false);
    if (node["repeat"]) {
      OMNI_RETURN_VAL_IF(!ValidateKeys(node, {"repeat", "as", "body", "when"}) ||
                             !node["repeat"].IsScalar() || !node["as"].IsScalar() ||
                             !ValidateSequence(node["body"], depth + 1U),
                         false);
    } else if (node["parallel"]) {
      OMNI_RETURN_VAL_IF(!ValidateKeys(node, {"parallel", "when"}) ||
                             !ValidateSequence(node["parallel"], depth + 1U),
                         false);
    } else if (node["body"]) {
      OMNI_RETURN_VAL_IF(!ValidateKeys(node, {"body", "when"}) ||
                             !ValidateSequence(node["body"], depth + 1U),
                         false);
    } else {
      OMNI_RETURN_VAL_IF(!node["op"].IsScalar(), false);
      const auto iterator = Operations().find(node["op"].as<std::string>());
      OMNI_RETURN_VAL_IF_LOG(iterator == Operations().end(), false, ERROR,
                             "Unknown graph operator: " << node["op"]);
      backend::ComputeOperation operation = iterator->second;
      std::set<std::string> fields = {"op", "when"};
      const bool is_valid = std::visit(
          [&node, &fields](auto &args) {
            return VisitFields(&args, [&node, &fields](const std::string &name, auto *const) {
              fields.insert(name);
              OMNI_RETURN_VAL_IF(!node[name], IsOptionalField(name));
              return node[name].IsScalar();
            });
          },
          operation);
      OMNI_RETURN_VAL_IF_LOG(!is_valid || !ValidateKeys(node, fields), false, ERROR,
                             "Invalid fields for graph operator: " << node["op"]);
    }
    OMNI_RETURN_VAL_IF(node["when"] && !node["when"].IsScalar(), false);
  }
  return true;
}

bool OperatorGraphConfig::Append(const std::string &section, const OperatorGraphBindings &bindings,
                                 const OperatorIterationBinder &bind_iteration,
                                 std::vector<backend::ComputeNode> *const nodes) {
  OMNI_RETURN_VAL_IF(nodes == nullptr, SetError("graph node output is null"));
  error_message_.clear();
  OMNI_RETURN_VAL_IF(!is_loaded_, SetError("operator graph configuration is not loaded"));
  std::vector<backend::ComputeNode> appended;
  std::vector<uint32_t> dependencies;
  try {
    OMNI_RETURN_VAL_IF(
        !ExpandSequence(document_[section], bindings, bind_iteration, 0U, &dependencies, &appended),
        SetError("failed to bind graph section: " + section));
  } catch (const YAML::Exception &error) {
    return SetError("graph section " + section + ": " + error.what());
  }
  const uint32_t offset = static_cast<uint32_t>(nodes->size());
  OMNI_RETURN_VAL_IF(nodes->size() + appended.size() > 16384U,
                     SetError("operator graph exceeds 16384 nodes"));
  std::vector<bool> has_successor(nodes->size(), false);
  for (const auto &node : *nodes) {
    for (const uint32_t dependency : node.dependencies) {
      OMNI_RETURN_VAL_IF(dependency >= offset, SetError("invalid existing graph dependency"));
      has_successor[dependency] = true;
    }
  }
  std::vector<uint32_t> previous_leaves;
  for (uint32_t index = 0U; index < offset; ++index) {
    if (!has_successor[index]) {
      previous_leaves.push_back(index);
    }
  }
  for (auto &node : appended) {
    for (auto &dependency : node.dependencies) {
      dependency += offset;
    }
    if (node.dependencies.empty()) {
      node.dependencies = previous_leaves;
    }
    nodes->push_back(std::move(node));
  }
  return true;
}

bool OperatorGraphConfig::ExpandSequence(const YAML::Node &sequence,
                                         const OperatorGraphBindings &bindings,
                                         const OperatorIterationBinder &bind_iteration,
                                         const size_t depth,
                                         std::vector<uint32_t> *const dependencies,
                                         std::vector<backend::ComputeNode> *const nodes) {
  OMNI_RETURN_VAL_IF(
      dependencies == nullptr || nodes == nullptr || depth > 16U || !sequence.IsSequence(), false);
  for (const auto &node : sequence) {
    if (node.IsSequence()) {
      OMNI_RETURN_VAL_IF(
          !ExpandSequence(node, bindings, bind_iteration, depth + 1U, dependencies, nodes), false);
      continue;
    }
    bool is_enabled = true;
    OMNI_RETURN_VAL_IF(node["when"] && !BindValue(node["when"], bindings, &is_enabled), false);
    if (!is_enabled) {
      continue;
    }
    if (node["repeat"]) {
      int32_t count = 0;
      OMNI_RETURN_VAL_IF(!BindValue(node["repeat"], bindings, &count) || count < 0 || count > 4096,
                         false);
      const std::string variable = node["as"].as<std::string>();
      for (int32_t index = 0; index < count; ++index) {
        OperatorGraphBindings iteration = bindings;
        iteration.values[variable] = index;
        OMNI_RETURN_VAL_IF(bind_iteration && !bind_iteration(variable, index, &iteration), false);
        OMNI_RETURN_VAL_IF(!ExpandSequence(node["body"], iteration, bind_iteration, depth + 1U,
                                           dependencies, nodes),
                           false);
      }
    } else if (node["parallel"]) {
      std::vector<uint32_t> branches;
      for (const auto &branch : node["parallel"]) {
        std::vector<uint32_t> branch_dependencies = *dependencies;
        YAML::Node body(YAML::NodeType::Sequence);
        body.push_back(branch);
        OMNI_RETURN_VAL_IF(!ExpandSequence(body, bindings, bind_iteration, depth + 1U,
                                           &branch_dependencies, nodes),
                           false);
        branches.insert(branches.end(), branch_dependencies.begin(), branch_dependencies.end());
      }
      *dependencies = std::move(branches);
    } else if (node["body"]) {
      OMNI_RETURN_VAL_IF(
          !ExpandSequence(node["body"], bindings, bind_iteration, depth + 1U, dependencies, nodes),
          false);
    } else {
      backend::ComputeOperation operation = Operations().at(node["op"].as<std::string>());
      const bool is_bound = std::visit(
          [&node, &bindings](auto &args) {
            return VisitFields(&args, [&node, &bindings](const std::string &name,
                                                         auto *const value) {
              if (!node[name] && IsOptionalField(name)) {
                return true;
              }
              OMNI_RETURN_VAL_IF_LOG(!BindValue(node[name], bindings, value), false, ERROR,
                                     "Cannot bind operator field: " << node["op"] << "." << name);
              if constexpr (std::is_same_v<std::decay_t<decltype(*value)>, backend::DeviceView>) {
                OMNI_RETURN_VAL_IF_LOG(!IsOptionalField(name) && !value->IsValid(), false, ERROR,
                                       "Invalid operator tensor: " << node["op"] << "." << name);
              }
              return true;
            });
          },
          operation);
      OMNI_RETURN_VAL_IF(!is_bound || nodes->size() >= 16384U, false);
      const uint32_t index = static_cast<uint32_t>(nodes->size());
      nodes->push_back({.operation = std::move(operation), .dependencies = *dependencies});
      *dependencies = {index};
    }
  }
  return true;
}

bool OperatorGraphConfig::SetError(const std::string &message) {
  error_message_ = message;
  LOG_ERROR(message);
  return false;
}

const std::string &OperatorGraphConfig::backend() const {
  return backend_;
}

const std::string &OperatorGraphConfig::error_message() const {
  return error_message_;
}

bool OperatorGraph::Prepare(const std::string &name,
                            const std::shared_ptr<backend::Backend> &backend,
                            const backend::ScalarType type,
                            const std::vector<backend::ComputeNode> &nodes) {
  error_message_.clear();
  is_prepared_ = false;
  OMNI_RETURN_VAL_IF_LOG(backend == nullptr,
                         (error_message_ = "operator graph backend is null", false), ERROR,
                         "Operator graph backend is null");
  std::vector<backend::ComputeOperation> operations;
  OMNI_RETURN_VAL_IF(!backend::CompileComputeGraph(type, backend->kind(), nodes, &operations),
                     (error_message_ = "failed to compile operator dependencies", false));
  OMNI_RETURN_VAL_IF(!graph_.Init(name, 1U), (error_message_ = graph_.error_message(), false));
  std::shared_ptr<graph::Element> previous;
  for (size_t index = 0U; index < operations.size(); ++index) {
    graph::NodeInfo info;
    info.name = name + "/op_" + std::to_string(index);
    if (previous != nullptr) {
      info.dependencies.push_back(previous);
    }
    previous = graph_.RegisterNode<ComputeOperatorNode>(info, backend, type, operations[index]);
    OMNI_RETURN_VAL_IF(previous == nullptr,
                       (error_message_ = "failed to register operator node: " + info.name, false));
  }
  OMNI_RETURN_VAL_IF(!graph_.Prepare(), (error_message_ = graph_.error_message(), false));
  is_prepared_ = true;
  return true;
}

bool OperatorGraph::Run() {
  OMNI_RETURN_VAL_IF_LOG(!is_prepared_, (error_message_ = "operator graph is not prepared", false),
                         ERROR, "Operator graph is not prepared");
  OMNI_RETURN_VAL_IF(!graph_.Process(),
                     (is_prepared_ = false, error_message_ = graph_.error_message(), false));
  return true;
}

const std::string &OperatorGraph::error_message() const {
  return error_message_;
}

} // namespace inference
} // namespace omni_runtime
