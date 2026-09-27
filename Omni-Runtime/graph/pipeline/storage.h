#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/graph/basic/status.h"
#include "Omni-Runtime/graph/element/element_define.h"

namespace omni_runtime {
namespace graph {

class Element;
class Pipeline;

class Storage final {
public:
  static Status Save(const Pipeline *const pipeline, const std::string &path);
  static Status Load(const std::string &path, Pipeline *const pipeline);

private:
  struct PassedParamRecord {
    std::string key;
    std::string type_name;
  };

  struct AspectRecord {
    std::string type_name;
    std::string param_type_name;
  };

  struct ParamRecord {
    std::string key;
    std::string type_name;
    bool enable_backtrace = false;
  };

  struct EventRecord {
    std::string key;
    std::string type_name;
    std::string param_type_name;
  };

  struct DaemonRecord {
    std::string type_name;
    std::string param_type_name;
    Milliseconds interval_ms = 0;
  };

  struct StageRecord {
    std::string key;
    std::string type_name;
    std::string param_type_name;
    int32_t threshold = 0;
  };

  struct ElementRecord {
    std::string type_name;
    std::string name;
    ElementType element_type = ElementType::kElement;
    std::size_t loop_count = 1U;
    Level level = 0;
    Milliseconds timeout_ms = 0;
    ElementTimeoutStrategy timeout_strategy = ElementTimeoutStrategy::kAsError;
    Index binding_index = -1;
    std::size_t parent_index = std::numeric_limits<std::size_t>::max();
    bool is_visible = true;
    bool is_macro = false;
    std::vector<std::size_t> dependency_indices;
    std::vector<AspectRecord> aspects;
    std::vector<PassedParamRecord> local_params;
    std::shared_ptr<Element> source;
  };

  struct PipelineRecord {
    uint32_t thread_count = 1U;
    EngineType engine_type = EngineType::kDynamic;
    std::vector<ParamRecord> params;
    std::vector<EventRecord> events;
    std::vector<DaemonRecord> daemons;
    std::vector<StageRecord> stages;
    std::vector<ElementRecord> elements;
  };

  static Status BuildRecord(const Pipeline *const pipeline, PipelineRecord *const record);
  static Status
  GatherElement(const std::shared_ptr<Element> &element, const std::size_t parent_index,
                std::unordered_map<const Element *, std::size_t> *const element_indices,
                std::vector<ElementRecord> *const records);
  static Status SaveRecord(const PipelineRecord &record, std::ostream *const stream);
  static Status LoadRecord(std::istream *const stream, PipelineRecord *const record);
  static Status RestoreRecord(const PipelineRecord &record, Pipeline *const pipeline);
};

} // namespace graph
} // namespace omni_runtime
