#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Omni-Runtime/graph/basic/status.h"
#include "Omni-Runtime/graph/basic/types.h"

namespace omni_runtime {
namespace graph {

class Element;

enum class ElementType : uint32_t {
  kElement = 0x00000000U,
  kNode = 0x00010000U,
  kGroup = 0x00020000U,
  kCluster = 0x00020001U,
  kRegion = 0x00020002U,
  kCondition = 0x00020004U,
  kSome = 0x00020008U,
  kMutable = 0x0002000AU,
  kMultiCondition = 0x00020014U,
  kAdapter = 0x00040000U,
  kFunction = 0x00040001U,
  kSingleton = 0x00040002U,
  kFence = 0x00040004U,
  kCoordinator = 0x00040008U,
};

enum class ElementState : uint8_t {
  kNormal = 0,
  kCancel = 1,
  kSuspend = 2,
  kTimeout = 3,
};

using PipelineState = ElementState;

enum class ElementTimeoutStrategy : uint8_t {
  kAsError = 0,
  kHoldByPipeline = 1,
  kNoHold = 2,
};

enum class EngineType : uint8_t {
  kDynamic = 0,
  kTopological = 1,
  kStatic = 2,
};

enum class FunctionType : uint8_t {
  kInit = 0,
  kRun = 1,
  kDestroy = 2,
};

enum class NodeType : uint8_t {
  kBasic = 0,
  kIo = 1,
  kCpu = 2,
  kGpu = 3,
};

enum class MultiConditionType : uint8_t {
  kSerial = 0,
  kParallel = 1,
};

struct ElementRelation {
  std::vector<std::shared_ptr<Element>> predecessors;
  std::vector<std::shared_ptr<Element>> successors;
  std::vector<std::shared_ptr<Element>> children;
  std::shared_ptr<Element> belong;
};

struct NodeInfo {
  std::string name;
  std::size_t loop_count = kDefaultLoopCount;
  std::vector<std::shared_ptr<Element>> dependencies;
};

using Task = std::function<Status()>;
using TaskGroup = std::vector<Task>;

} // namespace graph
} // namespace omni_runtime
