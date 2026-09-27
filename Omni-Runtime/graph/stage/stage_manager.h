#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "Omni-Runtime/utils/status_macros.h"
#include "Omni-Runtime/graph/stage/stage.h"

namespace omni_runtime {
namespace graph {

class Storage;

class StageManager final : public Object {
public:
  StageManager() = default;
  ~StageManager() override;

  template <typename TStage, typename TParam>
  Status Create(const std::string &key, const int32_t threshold, const TParam *const param) {
    static_assert(std::is_base_of<Stage, TStage>::value, "TStage must inherit Stage");
    static_assert(std::is_base_of<StageParam, TParam>::value, "TParam must inherit StageParam");
    OMNI_RETURN_VAL_IF_LOG(key.empty(), InvalidArgumentStatus("stage key is empty"), ERROR,
                      "stage key is empty");
    OMNI_RETURN_VAL_IF_LOG(param_manager_ == nullptr, InternalStatus("param manager is unavailable"),
                      ERROR, "param manager is unavailable");
    std::lock_guard<std::mutex> lock(mutex_);
    OMNI_RETURN_VAL_IF_LOG(stages_.find(key) != stages_.end(),
                      InvalidArgumentStatus("stage key is duplicated: " + key), ERROR,
                      "stage key is duplicated: " << key);
    auto stage = std::make_shared<TStage>();
    OMNI_RETURN_VAL_IF_LOG(!stage->Configure(threshold, param, param_manager_),
                      InvalidArgumentStatus("stage configuration is invalid"), ERROR,
                      "stage configuration is invalid");
    stages_.emplace(key, std::move(stage));
    return Status();
  }

  bool SetParamManager(const std::shared_ptr<ParamManager> &param_manager);
  Status WaitForReady(const std::string &key);
  Status Clear();

  Status Init() final;
  Status Destroy() final;

private:
  Status Run() final {
    return UnimplementedStatus("StageManager cannot run");
  }

  std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Stage>> stages_;
  std::shared_ptr<ParamManager> param_manager_;

  friend class Storage;
};

} // namespace graph
} // namespace omni_runtime
