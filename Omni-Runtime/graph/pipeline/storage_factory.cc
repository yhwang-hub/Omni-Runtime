#include "Omni-Runtime/graph/pipeline/storage_factory.h"

#include "Omni-Runtime/graph/element/element_include.h"
#include "Omni-Runtime/graph/param/passed_param.h"
#include "Omni-Runtime/graph/stage/stage.h"

namespace omni_runtime {
namespace graph {

std::mutex StorageFactory::mutex_;
std::unordered_map<std::string, std::function<std::shared_ptr<Object>()>> StorageFactory::creators_;

std::shared_ptr<Object> StorageFactory::Create(const std::string &type_name) {
  static_cast<void>(RegisterBuiltinTypes());
  std::function<std::shared_ptr<Object>()> creator;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = creators_.find(type_name);
    if (iter == creators_.end()) {
      return nullptr;
    }
    creator = iter->second;
  }
  return creator();
}

bool StorageFactory::Has(const std::string &type_name) {
  static_cast<void>(RegisterBuiltinTypes());
  std::lock_guard<std::mutex> lock(mutex_);
  return creators_.find(type_name) != creators_.end();
}

bool StorageFactory::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  creators_.clear();
  return true;
}

bool StorageFactory::RegisterBuiltinTypes() {
  return Register<DefaultPassedParam>() && Register<Aspect>() && Register<Stage>() &&
         Register<Function>() && Register<Fence>() && Register<Cluster>() && Register<Region>() &&
         Register<MultiCondition<MultiConditionType::kSerial>>() &&
         Register<MultiCondition<MultiConditionType::kParallel>>();
}

} // namespace graph
} // namespace omni_runtime
