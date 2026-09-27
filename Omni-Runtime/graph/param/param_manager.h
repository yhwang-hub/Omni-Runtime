#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/utils/logger.h"

#include "Omni-Runtime/graph/param/param.h"

namespace omni_runtime {
namespace graph {

class Storage;

class ParamManager final : public Object {
public:
  ParamManager() = default;
  ~ParamManager() override;

  template <typename T> Status Create(const std::string &key, const bool enable_backtrace) {
    static_assert(std::is_base_of<Param, T>::value, "T must inherit Param");
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = params_.find(key);
    if (iter != params_.end()) {
      OMNI_RETURN_VAL_IF_LOG(std::dynamic_pointer_cast<T>(iter->second) == nullptr,
                        InvalidArgumentStatus("param type conflicts for key " + key), ERROR,
                        "param type conflicts for key " << key);
      return Status();
    }

    auto param = std::make_shared<T>();
    param->key_ = key;
    param->enable_backtrace_ = enable_backtrace;
    params_.emplace(key, std::move(param));
    return Status();
  }

  template <typename T> std::shared_ptr<T> Get(const std::string &key) const {
    static_assert(std::is_base_of<Param, T>::value, "T must inherit Param");
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = params_.find(key);
    if (iter == params_.end()) {
      return nullptr;
    }
    return std::dynamic_pointer_cast<T>(iter->second);
  }

  Status Remove(const std::string &key);
  Status GetKeys(std::vector<std::string> *const keys) const;
  bool Has(const std::string &key) const;
  Status Setup();
  bool Reset(const Status &status);

  Status Init() final;
  Status Destroy() final;
  Status Clear();

private:
  Status Run() final {
    return UnimplementedStatus("ParamManager cannot run");
  }

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Param>> params_;

  friend class Storage;
};

} // namespace graph
} // namespace omni_runtime
