#include "Omni-Runtime/graph/pipeline/storage.h"

#include <fstream>
#include <istream>
#include <ostream>
#include <typeinfo>
#include <utility>

#include "Omni-Runtime/utils/logger.h"

#include "Omni-Runtime/utils/status_macros.h"
#include "Omni-Runtime/graph/element/element_include.h"
#include "Omni-Runtime/graph/pipeline/pipeline.h"
#include "Omni-Runtime/graph/pipeline/storage_factory.h"

namespace omni_runtime {
namespace graph {
namespace {

constexpr uint64_t kStorageMagic = 0x5A4F4D4E49475231ULL;
constexpr uint64_t kMaximumStorageItemCount = 1000000ULL;

template <typename T> bool WriteValue(const T &input, std::ostream *const stream) {
  OMNI_RETURN_VAL_IF(stream == nullptr, false);
  stream->write(reinterpret_cast<const char *>(&input), sizeof(T));
  return stream->good();
}

template <typename T> bool ReadValue(std::istream *const stream, T *const output) {
  OMNI_RETURN_VAL_IF(stream == nullptr || output == nullptr, false);
  stream->read(reinterpret_cast<char *>(output), sizeof(T));
  return stream->good();
}

bool WriteString(const std::string &input, std::ostream *const stream) {
  OMNI_RETURN_VAL_IF(stream == nullptr, false);
  const uint64_t size = static_cast<uint64_t>(input.size());
  OMNI_RETURN_VAL_IF(!WriteValue(size, stream), false);
  stream->write(input.data(), static_cast<std::streamsize>(size));
  return stream->good();
}

bool ReadString(std::istream *const stream, std::string *const output) {
  OMNI_RETURN_VAL_IF(stream == nullptr || output == nullptr, false);
  uint64_t size = 0U;
  OMNI_RETURN_VAL_IF(!ReadValue(stream, &size) || size > kMaximumStorageItemCount, false);
  output->resize(static_cast<std::size_t>(size));
  stream->read(output->data(), static_cast<std::streamsize>(size));
  return stream->good();
}

template <typename T> std::string TypeName(const std::shared_ptr<T> &object) {
  if (object == nullptr) {
    return {};
  }
  const T *const raw_object = object.get();
  return typeid(*raw_object).name();
}

std::shared_ptr<PassedParam> CreatePassedParam(const std::string &type_name) {
  if (type_name.empty()) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<PassedParam>(StorageFactory::Create(type_name));
}

} // namespace

Status Storage::Save(const Pipeline *const pipeline, const std::string &path) {
  OMNI_RETURN_VAL_IF_LOG(pipeline == nullptr, InvalidArgumentStatus("pipeline is null"), ERROR,
                    "pipeline is null");
  OMNI_RETURN_VAL_IF_LOG(path.empty(), InvalidArgumentStatus("storage path is empty"), ERROR,
                    "storage path is empty");
  PipelineRecord record;
  OMNI_RETURN_STATUS_IF_NOT_OK(BuildRecord(pipeline, &record));
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  OMNI_RETURN_VAL_IF_LOG(!stream.is_open(), InternalStatus("failed to open storage path"), ERROR,
                    "failed to open storage path: " << path);
  return SaveRecord(record, &stream);
}

Status Storage::Load(const std::string &path, Pipeline *const pipeline) {
  OMNI_RETURN_VAL_IF_LOG(pipeline == nullptr, InvalidArgumentStatus("pipeline is null"), ERROR,
                    "pipeline is null");
  OMNI_RETURN_VAL_IF_LOG(path.empty(), InvalidArgumentStatus("storage path is empty"), ERROR,
                    "storage path is empty");
  std::ifstream stream(path, std::ios::binary);
  OMNI_RETURN_VAL_IF_LOG(!stream.is_open(), NotFoundStatus("failed to open storage path"), ERROR,
                    "failed to open storage path: " << path);
  PipelineRecord record;
  OMNI_RETURN_STATUS_IF_NOT_OK(LoadRecord(&stream, &record));
  return RestoreRecord(record, pipeline);
}

Status Storage::BuildRecord(const Pipeline *const pipeline, PipelineRecord *const record) {
  OMNI_RETURN_VAL_IF_LOG(pipeline == nullptr || record == nullptr,
                    InvalidArgumentStatus("storage input is null"), ERROR, "storage input is null");
  OMNI_RETURN_VAL_IF_LOG(pipeline->is_initialized_,
                    InvalidArgumentStatus("initialized pipeline cannot be saved"), ERROR,
                    "initialized pipeline cannot be saved");
  record->thread_count = pipeline->config_.thread_count;
  record->engine_type = pipeline->element_manager_->engine_type();
  record->params.clear();
  record->events.clear();
  record->daemons.clear();
  record->stages.clear();
  record->elements.clear();

  {
    std::lock_guard<std::mutex> lock(pipeline->param_manager_->mutex_);
    for (const auto &item : pipeline->param_manager_->params_) {
      ParamRecord param_record;
      param_record.key = item.first;
      param_record.type_name = TypeName(item.second);
      param_record.enable_backtrace = item.second->enable_backtrace_;
      record->params.emplace_back(std::move(param_record));
    }
  }
  {
    std::lock_guard<std::mutex> lock(pipeline->event_manager_->mutex_);
    for (const auto &item : pipeline->event_manager_->events_) {
      EventRecord event_record;
      event_record.key = item.first;
      event_record.type_name = TypeName(item.second);
      event_record.param_type_name = TypeName(item.second->param_);
      record->events.emplace_back(std::move(event_record));
    }
  }
  {
    std::lock_guard<std::mutex> lock(pipeline->daemon_manager_->mutex_);
    for (const auto &daemon : pipeline->daemon_manager_->daemons_) {
      DaemonRecord daemon_record;
      daemon_record.type_name = TypeName(daemon);
      daemon_record.param_type_name = TypeName(daemon->param_);
      daemon_record.interval_ms = daemon->interval_ms();
      record->daemons.emplace_back(std::move(daemon_record));
    }
  }
  {
    std::lock_guard<std::mutex> lock(pipeline->stage_manager_->mutex_);
    for (const auto &item : pipeline->stage_manager_->stages_) {
      StageRecord stage_record;
      stage_record.key = item.first;
      stage_record.type_name = TypeName(item.second);
      stage_record.param_type_name = TypeName(item.second->param_);
      stage_record.threshold = item.second->threshold();
      record->stages.emplace_back(std::move(stage_record));
    }
  }

  std::vector<std::shared_ptr<Element>> top_level_elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(pipeline->element_manager_->GetElements(&top_level_elements));
  std::unordered_map<const Element *, std::size_t> element_indices;
  for (const auto &element : top_level_elements) {
    OMNI_RETURN_STATUS_IF_NOT_OK(GatherElement(element, std::numeric_limits<std::size_t>::max(),
                                          &element_indices, &record->elements));
  }
  for (auto &element_record : record->elements) {
    ElementRelation relation;
    OMNI_RETURN_STATUS_IF_NOT_OK(element_record.source->GetRelation(&relation));
    element_record.dependency_indices.clear();
    for (const auto &dependency : relation.predecessors) {
      const auto iter = element_indices.find(dependency.get());
      OMNI_RETURN_VAL_IF_LOG(iter == element_indices.end(),
                        InvalidArgumentStatus("dependency is outside saved pipeline"), ERROR,
                        "dependency is outside saved pipeline");
      element_record.dependency_indices.emplace_back(iter->second);
    }
  }
  return Status();
}

Status
Storage::GatherElement(const std::shared_ptr<Element> &element, const std::size_t parent_index,
                       std::unordered_map<const Element *, std::size_t> *const element_indices,
                       std::vector<ElementRecord> *const records) {
  OMNI_RETURN_VAL_IF_LOG(element == nullptr || element_indices == nullptr || records == nullptr,
                    InvalidArgumentStatus("element storage input is null"), ERROR,
                    "element storage input is null");
  OMNI_RETURN_VAL_IF_LOG(element_indices->find(element.get()) != element_indices->end(),
                    InvalidArgumentStatus("element occurs more than once"), ERROR,
                    "element occurs more than once: " << element->name());
  ElementRecord record;
  const Element *const raw_element = element.get();
  record.type_name = typeid(*raw_element).name();
  record.name = element->name();
  record.element_type = element->element_type();
  record.loop_count = element->loop_count();
  record.level = element->level();
  record.timeout_ms = element->timeout_ms();
  record.timeout_strategy = element->timeout_strategy();
  record.binding_index = element->binding_index();
  record.parent_index = parent_index;
  record.is_visible = element->is_visible();
  record.is_macro = element->is_macro();
  record.source = element;
  if (element->aspect_manager_ != nullptr) {
    std::lock_guard<std::mutex> lock(element->aspect_manager_->mutex_);
    for (const auto &aspect : element->aspect_manager_->aspects_) {
      AspectRecord aspect_record;
      aspect_record.type_name = TypeName(aspect);
      aspect_record.param_type_name = TypeName(aspect->param_);
      record.aspects.emplace_back(std::move(aspect_record));
    }
  }
  {
    std::lock_guard<std::mutex> lock(element->local_param_mutex_);
    for (const auto &item : element->local_params_) {
      PassedParamRecord local_param_record;
      local_param_record.key = item.first;
      local_param_record.type_name = TypeName(item.second);
      record.local_params.emplace_back(std::move(local_param_record));
    }
  }
  const std::size_t current_index = records->size();
  element_indices->emplace(element.get(), current_index);
  records->emplace_back(std::move(record));

  ElementRelation relation;
  OMNI_RETURN_STATUS_IF_NOT_OK(element->GetRelation(&relation));
  for (const auto &child : relation.children) {
    OMNI_RETURN_STATUS_IF_NOT_OK(GatherElement(child, current_index, element_indices, records));
  }
  return Status();
}

Status Storage::SaveRecord(const PipelineRecord &record, std::ostream *const stream) {
  OMNI_RETURN_VAL_IF_LOG(stream == nullptr, InvalidArgumentStatus("storage stream is null"), ERROR,
                    "storage stream is null");
  const uint8_t engine_type = static_cast<uint8_t>(record.engine_type);
  const uint64_t param_count = static_cast<uint64_t>(record.params.size());
  const uint64_t event_count = static_cast<uint64_t>(record.events.size());
  const uint64_t daemon_count = static_cast<uint64_t>(record.daemons.size());
  const uint64_t stage_count = static_cast<uint64_t>(record.stages.size());
  const uint64_t element_count = static_cast<uint64_t>(record.elements.size());
  OMNI_RETURN_VAL_IF_LOG(
      !WriteValue(kStorageMagic, stream) || !WriteValue(record.thread_count, stream) ||
          !WriteValue(engine_type, stream) || !WriteValue(param_count, stream) ||
          !WriteValue(event_count, stream) || !WriteValue(daemon_count, stream) ||
          !WriteValue(stage_count, stream) || !WriteValue(element_count, stream),
      InternalStatus("failed to write storage header"), ERROR, "failed to write storage header");
  for (const auto &param : record.params) {
    const uint8_t enable_backtrace = param.enable_backtrace ? 1U : 0U;
    OMNI_RETURN_VAL_IF_LOG(!WriteString(param.key, stream) || !WriteString(param.type_name, stream) ||
                          !WriteValue(enable_backtrace, stream),
                      InternalStatus("failed to write param storage"), ERROR,
                      "failed to write param storage: " << param.key);
  }
  for (const auto &event : record.events) {
    OMNI_RETURN_VAL_IF_LOG(!WriteString(event.key, stream) || !WriteString(event.type_name, stream) ||
                          !WriteString(event.param_type_name, stream),
                      InternalStatus("failed to write event storage"), ERROR,
                      "failed to write event storage: " << event.key);
  }
  for (const auto &daemon : record.daemons) {
    OMNI_RETURN_VAL_IF_LOG(
        !WriteString(daemon.type_name, stream) || !WriteString(daemon.param_type_name, stream) ||
            !WriteValue(daemon.interval_ms, stream),
        InternalStatus("failed to write daemon storage"), ERROR, "failed to write daemon storage");
  }
  for (const auto &stage : record.stages) {
    OMNI_RETURN_VAL_IF_LOG(!WriteString(stage.key, stream) || !WriteString(stage.type_name, stream) ||
                          !WriteString(stage.param_type_name, stream) ||
                          !WriteValue(stage.threshold, stream),
                      InternalStatus("failed to write stage storage"), ERROR,
                      "failed to write stage storage: " << stage.key);
  }
  for (const auto &element : record.elements) {
    const uint32_t element_type = static_cast<uint32_t>(element.element_type);
    const uint64_t loop_count = static_cast<uint64_t>(element.loop_count);
    const uint8_t timeout_strategy = static_cast<uint8_t>(element.timeout_strategy);
    const uint64_t parent_index = static_cast<uint64_t>(element.parent_index);
    const uint8_t is_visible = element.is_visible ? 1U : 0U;
    const uint8_t is_macro = element.is_macro ? 1U : 0U;
    const uint64_t dependency_count = static_cast<uint64_t>(element.dependency_indices.size());
    const uint64_t aspect_count = static_cast<uint64_t>(element.aspects.size());
    const uint64_t local_param_count = static_cast<uint64_t>(element.local_params.size());
    OMNI_RETURN_VAL_IF_LOG(
        !WriteString(element.type_name, stream) || !WriteString(element.name, stream) ||
            !WriteValue(element_type, stream) || !WriteValue(loop_count, stream) ||
            !WriteValue(element.level, stream) || !WriteValue(element.timeout_ms, stream) ||
            !WriteValue(timeout_strategy, stream) || !WriteValue(element.binding_index, stream) ||
            !WriteValue(parent_index, stream) || !WriteValue(is_visible, stream) ||
            !WriteValue(is_macro, stream) || !WriteValue(dependency_count, stream) ||
            !WriteValue(aspect_count, stream) || !WriteValue(local_param_count, stream),
        InternalStatus("failed to write element storage"), ERROR,
        "failed to write element storage: " << element.name);
    for (const std::size_t dependency_index : element.dependency_indices) {
      const uint64_t serialized_index = static_cast<uint64_t>(dependency_index);
      OMNI_RETURN_VAL_IF_LOG(!WriteValue(serialized_index, stream),
                        InternalStatus("failed to write dependency storage"), ERROR,
                        "failed to write dependency storage");
    }
    for (const auto &aspect : element.aspects) {
      OMNI_RETURN_VAL_IF_LOG(!WriteString(aspect.type_name, stream) ||
                            !WriteString(aspect.param_type_name, stream),
                        InternalStatus("failed to write aspect storage"), ERROR,
                        "failed to write aspect storage");
    }
    for (const auto &local_param : element.local_params) {
      OMNI_RETURN_VAL_IF_LOG(!WriteString(local_param.key, stream) ||
                            !WriteString(local_param.type_name, stream),
                        InternalStatus("failed to write local param storage"), ERROR,
                        "failed to write local param storage: " << local_param.key);
    }
  }
  return Status();
}

Status Storage::LoadRecord(std::istream *const stream, PipelineRecord *const record) {
  OMNI_RETURN_VAL_IF_LOG(stream == nullptr || record == nullptr,
                    InvalidArgumentStatus("storage input is null"), ERROR, "storage input is null");
  uint64_t magic = 0U;
  uint8_t engine_type = 0U;
  uint64_t param_count = 0U;
  uint64_t event_count = 0U;
  uint64_t daemon_count = 0U;
  uint64_t stage_count = 0U;
  uint64_t element_count = 0U;
  OMNI_RETURN_VAL_IF_LOG(
      !ReadValue(stream, &magic) || magic != kStorageMagic ||
          !ReadValue(stream, &record->thread_count) || !ReadValue(stream, &engine_type) ||
          !ReadValue(stream, &param_count) || !ReadValue(stream, &event_count) ||
          !ReadValue(stream, &daemon_count) || !ReadValue(stream, &stage_count) ||
          !ReadValue(stream, &element_count) || param_count > kMaximumStorageItemCount ||
          event_count > kMaximumStorageItemCount || daemon_count > kMaximumStorageItemCount ||
          stage_count > kMaximumStorageItemCount || element_count > kMaximumStorageItemCount,
      InvalidArgumentStatus("invalid storage header"), ERROR, "invalid storage header");
  OMNI_RETURN_VAL_IF_LOG(engine_type > static_cast<uint8_t>(EngineType::kStatic),
                    InvalidArgumentStatus("invalid stored engine type"), ERROR,
                    "invalid stored engine type");
  record->engine_type = static_cast<EngineType>(engine_type);
  record->params.clear();
  record->params.resize(static_cast<std::size_t>(param_count));
  for (auto &param : record->params) {
    uint8_t enable_backtrace = 0U;
    OMNI_RETURN_VAL_IF_LOG(!ReadString(stream, &param.key) || !ReadString(stream, &param.type_name) ||
                          !ReadValue(stream, &enable_backtrace),
                      InvalidArgumentStatus("invalid param storage"), ERROR,
                      "invalid param storage");
    param.enable_backtrace = enable_backtrace != 0U;
  }
  record->events.clear();
  record->events.resize(static_cast<std::size_t>(event_count));
  for (auto &event : record->events) {
    OMNI_RETURN_VAL_IF_LOG(!ReadString(stream, &event.key) || !ReadString(stream, &event.type_name) ||
                          !ReadString(stream, &event.param_type_name),
                      InvalidArgumentStatus("invalid event storage"), ERROR,
                      "invalid event storage");
  }
  record->daemons.clear();
  record->daemons.resize(static_cast<std::size_t>(daemon_count));
  for (auto &daemon : record->daemons) {
    OMNI_RETURN_VAL_IF_LOG(
        !ReadString(stream, &daemon.type_name) || !ReadString(stream, &daemon.param_type_name) ||
            !ReadValue(stream, &daemon.interval_ms),
        InvalidArgumentStatus("invalid daemon storage"), ERROR, "invalid daemon storage");
  }
  record->stages.clear();
  record->stages.resize(static_cast<std::size_t>(stage_count));
  for (auto &stage : record->stages) {
    OMNI_RETURN_VAL_IF_LOG(
        !ReadString(stream, &stage.key) || !ReadString(stream, &stage.type_name) ||
            !ReadString(stream, &stage.param_type_name) || !ReadValue(stream, &stage.threshold),
        InvalidArgumentStatus("invalid stage storage"), ERROR, "invalid stage storage");
  }
  record->elements.clear();
  record->elements.resize(static_cast<std::size_t>(element_count));
  for (auto &element : record->elements) {
    uint32_t element_type = 0U;
    uint64_t loop_count = 0U;
    uint8_t timeout_strategy = 0U;
    uint64_t parent_index = 0U;
    uint8_t is_visible = 0U;
    uint8_t is_macro = 0U;
    uint64_t dependency_count = 0U;
    uint64_t aspect_count = 0U;
    uint64_t local_param_count = 0U;
    OMNI_RETURN_VAL_IF_LOG(
        !ReadString(stream, &element.type_name) || !ReadString(stream, &element.name) ||
            !ReadValue(stream, &element_type) || !ReadValue(stream, &loop_count) ||
            !ReadValue(stream, &element.level) || !ReadValue(stream, &element.timeout_ms) ||
            !ReadValue(stream, &timeout_strategy) || !ReadValue(stream, &element.binding_index) ||
            !ReadValue(stream, &parent_index) || !ReadValue(stream, &is_visible) ||
            !ReadValue(stream, &is_macro) || !ReadValue(stream, &dependency_count) ||
            !ReadValue(stream, &aspect_count) || !ReadValue(stream, &local_param_count) ||
            dependency_count > kMaximumStorageItemCount ||
            aspect_count > kMaximumStorageItemCount || local_param_count > kMaximumStorageItemCount,
        InvalidArgumentStatus("invalid element storage"), ERROR, "invalid element storage");
    element.element_type = static_cast<ElementType>(element_type);
    element.loop_count = static_cast<std::size_t>(loop_count);
    element.timeout_strategy = static_cast<ElementTimeoutStrategy>(timeout_strategy);
    element.parent_index = static_cast<std::size_t>(parent_index);
    element.is_visible = is_visible != 0U;
    element.is_macro = is_macro != 0U;
    element.dependency_indices.resize(static_cast<std::size_t>(dependency_count));
    for (std::size_t &dependency_index : element.dependency_indices) {
      uint64_t serialized_index = 0U;
      OMNI_RETURN_VAL_IF_LOG(!ReadValue(stream, &serialized_index),
                        InvalidArgumentStatus("invalid dependency storage"), ERROR,
                        "invalid dependency storage");
      dependency_index = static_cast<std::size_t>(serialized_index);
    }
    element.aspects.resize(static_cast<std::size_t>(aspect_count));
    for (auto &aspect : element.aspects) {
      OMNI_RETURN_VAL_IF_LOG(
          !ReadString(stream, &aspect.type_name) || !ReadString(stream, &aspect.param_type_name),
          InvalidArgumentStatus("invalid aspect storage"), ERROR, "invalid aspect storage");
    }
    element.local_params.resize(static_cast<std::size_t>(local_param_count));
    for (auto &local_param : element.local_params) {
      OMNI_RETURN_VAL_IF_LOG(!ReadString(stream, &local_param.key) ||
                            !ReadString(stream, &local_param.type_name),
                        InvalidArgumentStatus("invalid local param storage"), ERROR,
                        "invalid local param storage");
    }
  }
  return Status();
}

Status Storage::RestoreRecord(const PipelineRecord &record, Pipeline *const pipeline) {
  OMNI_RETURN_VAL_IF_LOG(pipeline == nullptr, InvalidArgumentStatus("pipeline is null"), ERROR,
                    "pipeline is null");
  OMNI_RETURN_VAL_IF_LOG(pipeline->is_initialized_,
                    InvalidArgumentStatus("initialized pipeline cannot be loaded"), ERROR,
                    "initialized pipeline cannot be loaded");
  std::vector<std::shared_ptr<Element>> existing_elements;
  OMNI_RETURN_STATUS_IF_NOT_OK(pipeline->element_manager_->GetElements(&existing_elements));
  OMNI_RETURN_VAL_IF_LOG(!existing_elements.empty(),
                    InvalidArgumentStatus("pipeline must be empty before loading"), ERROR,
                    "pipeline must be empty before loading");
  OMNI_RETURN_VAL_IF_LOG(record.thread_count == 0U, InvalidArgumentStatus("stored thread count is zero"),
                    ERROR, "stored thread count is zero");
  {
    std::lock_guard<std::mutex> lock(pipeline->param_manager_->mutex_);
    OMNI_RETURN_VAL_IF_LOG(!pipeline->param_manager_->params_.empty(),
                      InvalidArgumentStatus("pipeline params must be empty before loading"), ERROR,
                      "pipeline params must be empty before loading");
  }
  {
    std::lock_guard<std::mutex> lock(pipeline->event_manager_->mutex_);
    OMNI_RETURN_VAL_IF_LOG(!pipeline->event_manager_->events_.empty(),
                      InvalidArgumentStatus("pipeline events must be empty before loading"), ERROR,
                      "pipeline events must be empty before loading");
  }
  {
    std::lock_guard<std::mutex> lock(pipeline->daemon_manager_->mutex_);
    OMNI_RETURN_VAL_IF_LOG(!pipeline->daemon_manager_->daemons_.empty(),
                      InvalidArgumentStatus("pipeline daemons must be empty before loading"), ERROR,
                      "pipeline daemons must be empty before loading");
  }
  {
    std::lock_guard<std::mutex> lock(pipeline->stage_manager_->mutex_);
    OMNI_RETURN_VAL_IF_LOG(!pipeline->stage_manager_->stages_.empty(),
                      InvalidArgumentStatus("pipeline stages must be empty before loading"), ERROR,
                      "pipeline stages must be empty before loading");
  }

  std::vector<std::shared_ptr<Param>> params;
  params.reserve(record.params.size());
  for (const auto &param_record : record.params) {
    const auto param =
        std::dynamic_pointer_cast<Param>(StorageFactory::Create(param_record.type_name));
    OMNI_RETURN_VAL_IF_LOG(param == nullptr,
                      NotFoundStatus("param type is not registered: " + param_record.type_name),
                      ERROR, "param type is not registered: " << param_record.type_name);
    param->key_ = param_record.key;
    param->enable_backtrace_ = param_record.enable_backtrace;
    params.emplace_back(param);
  }

  std::vector<std::pair<std::string, std::shared_ptr<Event>>> events;
  events.reserve(record.events.size());
  for (const auto &event_record : record.events) {
    const auto event =
        std::dynamic_pointer_cast<Event>(StorageFactory::Create(event_record.type_name));
    OMNI_RETURN_VAL_IF_LOG(event == nullptr,
                      NotFoundStatus("event type is not registered: " + event_record.type_name),
                      ERROR, "event type is not registered: " << event_record.type_name);
    const auto passed_param = CreatePassedParam(event_record.param_type_name);
    OMNI_RETURN_VAL_IF_LOG(!event_record.param_type_name.empty() && passed_param == nullptr,
                      NotFoundStatus("event param type is not registered"), ERROR,
                      "event param type is not registered: " << event_record.param_type_name);
    OMNI_RETURN_VAL_IF_LOG(!event->SetParam(passed_param.get()),
                      InternalStatus("failed to restore event param"), ERROR,
                      "failed to restore event param: " << event_record.key);
    events.emplace_back(event_record.key, event);
  }

  std::vector<std::shared_ptr<Daemon>> daemons;
  daemons.reserve(record.daemons.size());
  for (const auto &daemon_record : record.daemons) {
    const auto daemon =
        std::dynamic_pointer_cast<Daemon>(StorageFactory::Create(daemon_record.type_name));
    OMNI_RETURN_VAL_IF_LOG(daemon == nullptr,
                      NotFoundStatus("daemon type is not registered: " + daemon_record.type_name),
                      ERROR, "daemon type is not registered: " << daemon_record.type_name);
    const auto passed_param = CreatePassedParam(daemon_record.param_type_name);
    OMNI_RETURN_VAL_IF_LOG(!daemon_record.param_type_name.empty() && passed_param == nullptr,
                      NotFoundStatus("daemon param type is not registered"), ERROR,
                      "daemon param type is not registered: " << daemon_record.param_type_name);
    OMNI_RETURN_VAL_IF_LOG(!daemon->set_interval_ms(daemon_record.interval_ms) ||
                          !daemon->SetParam(passed_param.get()),
                      InvalidArgumentStatus("failed to restore daemon"), ERROR,
                      "failed to restore daemon");
    daemons.emplace_back(daemon);
  }

  std::vector<std::pair<std::string, std::shared_ptr<Stage>>> stages;
  stages.reserve(record.stages.size());
  for (const auto &stage_record : record.stages) {
    const auto stage =
        std::dynamic_pointer_cast<Stage>(StorageFactory::Create(stage_record.type_name));
    OMNI_RETURN_VAL_IF_LOG(stage == nullptr,
                      NotFoundStatus("stage type is not registered: " + stage_record.type_name),
                      ERROR, "stage type is not registered: " << stage_record.type_name);
    const auto passed_param = CreatePassedParam(stage_record.param_type_name);
    OMNI_RETURN_VAL_IF_LOG(!stage_record.param_type_name.empty() && passed_param == nullptr,
                      NotFoundStatus("stage param type is not registered"), ERROR,
                      "stage param type is not registered: " << stage_record.param_type_name);
    OMNI_RETURN_VAL_IF_LOG(
        !stage->Configure(stage_record.threshold, passed_param.get(), pipeline->param_manager_),
        InvalidArgumentStatus("failed to restore stage"), ERROR,
        "failed to restore stage: " << stage_record.key);
    stages.emplace_back(stage_record.key, stage);
  }

  std::vector<std::shared_ptr<Element>> elements;
  elements.reserve(record.elements.size());
  for (const auto &element_record : record.elements) {
    const std::shared_ptr<Object> object = StorageFactory::Create(element_record.type_name);
    const auto element = std::dynamic_pointer_cast<Element>(object);
    OMNI_RETURN_VAL_IF_LOG(element == nullptr,
                      NotFoundStatus("element type is not registered: " + element_record.type_name),
                      ERROR, "element type is not registered: " << element_record.type_name);
    OMNI_RETURN_VAL_IF_LOG(element->element_type() != element_record.element_type,
                      InvalidArgumentStatus("stored element type conflicts"), ERROR,
                      "stored element type conflicts: " << element_record.name);
    OMNI_RETURN_VAL_IF_LOG(
        !element->set_name(element_record.name) ||
            !element->set_loop_count(element_record.loop_count) ||
            !element->set_level(element_record.level) ||
            !element->set_visible(element_record.is_visible) ||
            !element->set_timeout(element_record.timeout_ms, element_record.timeout_strategy) ||
            !element->set_binding_index(element_record.binding_index) ||
            (element_record.is_macro && !element->set_macro(true)),
        InvalidArgumentStatus("failed to restore element configuration"), ERROR,
        "failed to restore element configuration: " << element_record.name);
    for (const auto &aspect_record : element_record.aspects) {
      const auto aspect =
          std::dynamic_pointer_cast<Aspect>(StorageFactory::Create(aspect_record.type_name));
      OMNI_RETURN_VAL_IF_LOG(aspect == nullptr,
                        NotFoundStatus("aspect type is not registered: " + aspect_record.type_name),
                        ERROR, "aspect type is not registered: " << aspect_record.type_name);
      const auto passed_param = CreatePassedParam(aspect_record.param_type_name);
      OMNI_RETURN_VAL_IF_LOG(!aspect_record.param_type_name.empty() && passed_param == nullptr,
                        NotFoundStatus("aspect param type is not registered"), ERROR,
                        "aspect param type is not registered: " << aspect_record.param_type_name);
      OMNI_RETURN_STATUS_IF_NOT_OK(element->AddAspect(aspect, passed_param.get()));
    }
    for (const auto &local_param_record : element_record.local_params) {
      const auto passed_param = CreatePassedParam(local_param_record.type_name);
      OMNI_RETURN_VAL_IF_LOG(passed_param == nullptr,
                        NotFoundStatus("local param type is not registered"), ERROR,
                        "local param type is not registered: " << local_param_record.type_name);
      element->local_params_[local_param_record.key] = passed_param;
    }
    elements.emplace_back(element);
  }

  for (std::size_t element_index = 0U; element_index < record.elements.size(); ++element_index) {
    const std::size_t parent_index = record.elements[element_index].parent_index;
    if (parent_index == std::numeric_limits<std::size_t>::max()) {
      continue;
    }
    OMNI_RETURN_VAL_IF_LOG(parent_index >= elements.size(),
                      InvalidArgumentStatus("stored parent index is invalid"), ERROR,
                      "stored parent index is invalid");
    const auto parent = std::dynamic_pointer_cast<Group>(elements[parent_index]);
    OMNI_RETURN_VAL_IF_LOG(parent == nullptr, InvalidArgumentStatus("stored parent is not a group"),
                      ERROR, "stored parent is not a group");
    OMNI_RETURN_STATUS_IF_NOT_OK(parent->AddElement(elements[element_index]));
  }
  for (std::size_t element_index = 0U; element_index < record.elements.size(); ++element_index) {
    for (const std::size_t dependency_index : record.elements[element_index].dependency_indices) {
      OMNI_RETURN_VAL_IF_LOG(dependency_index >= elements.size(),
                        InvalidArgumentStatus("stored dependency index is invalid"), ERROR,
                        "stored dependency index is invalid");
      OMNI_RETURN_STATUS_IF_NOT_OK(elements[element_index]->AddDependency(elements[dependency_index]));
    }
  }

  pipeline->config_.thread_count = record.thread_count;
  OMNI_RETURN_VAL_IF_LOG(!pipeline->element_manager_->set_engine_type(record.engine_type),
                    InternalStatus("failed to restore engine type"), ERROR,
                    "failed to restore engine type");
  {
    std::lock_guard<std::mutex> lock(pipeline->param_manager_->mutex_);
    for (const auto &param : params) {
      pipeline->param_manager_->params_.emplace(param->key_, param);
    }
  }
  {
    std::lock_guard<std::mutex> lock(pipeline->event_manager_->mutex_);
    for (const auto &item : events) {
      pipeline->event_manager_->events_.emplace(item.first, item.second);
    }
  }
  for (const auto &daemon : daemons) {
    OMNI_RETURN_STATUS_IF_NOT_OK(pipeline->daemon_manager_->Add(daemon));
  }
  {
    std::lock_guard<std::mutex> lock(pipeline->stage_manager_->mutex_);
    for (const auto &item : stages) {
      pipeline->stage_manager_->stages_.emplace(item.first, item.second);
    }
  }
  for (std::size_t element_index = 0U; element_index < record.elements.size(); ++element_index) {
    if (record.elements[element_index].parent_index == std::numeric_limits<std::size_t>::max()) {
      OMNI_RETURN_STATUS_IF_NOT_OK(pipeline->element_manager_->Add(elements[element_index]));
    }
  }
  return pipeline->element_manager_->Validate();
}

} // namespace graph
} // namespace omni_runtime
