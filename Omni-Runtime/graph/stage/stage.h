#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

#include "Omni-Runtime/utils/logger.h"

#include "Omni-Runtime/graph/basic/descriptor.h"
#include "Omni-Runtime/graph/basic/object.h"
#include "Omni-Runtime/graph/param/param_manager.h"
#include "Omni-Runtime/graph/param/passed_param.h"

namespace omni_runtime {
namespace graph {

class Storage;

class Stage : public Object, public Descriptor {
public:
  Stage() : Descriptor("stage") {
  }
  ~Stage() override = default;

  int32_t threshold() const;

protected:
  virtual bool Launch(const StageParam *const param) {
    static_cast<void>(param);
    return true;
  }

  template <typename T> std::shared_ptr<T> GetParam(const std::string &key) const {
    const auto manager = param_manager_.lock();
    OMNI_RETURN_VAL_IF_LOG(manager == nullptr, nullptr, ERROR, "param manager is unavailable");
    const auto param = manager->Get<T>(key);
    if (param != nullptr) {
      static_cast<void>(param->AddBacktrace(name()));
    }
    return param;
  }

private:
  Status Run() final {
    return UnimplementedStatus("Stage cannot run directly");
  }

  bool Configure(const int32_t threshold, const StageParam *const param,
                 const std::shared_ptr<ParamManager> &param_manager);
  Status Wait();

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::weak_ptr<ParamManager> param_manager_;
  std::shared_ptr<StageParam> param_;
  int32_t threshold_ = 0;
  int32_t current_count_ = 0;
  uint64_t generation_ = 0U;

  friend class StageManager;
  friend class Storage;
};

} // namespace graph
} // namespace omni_runtime
