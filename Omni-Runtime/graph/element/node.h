#pragma once

#include <tuple>
#include <utility>

#include "Omni-Runtime/graph/element/element.h"

namespace omni_runtime {
namespace graph {

class Node : public Element {
public:
  Node();
  ~Node() override = default;

  bool set_node_type(const NodeType node_type) {
    node_type_ = node_type;
    return true;
  }

  NodeType node_type() const {
    return node_type_;
  }

protected:
  Status SpawnTasks(const TaskGroup &tasks, const Milliseconds timeout_ms) const {
    return Spawn(tasks, timeout_ms);
  }

private:
  NodeType node_type_ = NodeType::kBasic;
};

template <typename... Args> class TemplateNode : public Node {
public:
  explicit TemplateNode(const Args &...args) : args_(args...) {
  }

protected:
  std::tuple<Args...> &args() {
    return args_;
  }

private:
  std::tuple<Args...> args_;
};

} // namespace graph
} // namespace omni_runtime
