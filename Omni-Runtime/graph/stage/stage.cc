#include "Omni-Runtime/graph/stage/stage.h"

namespace omni_runtime {
namespace graph {

int32_t Stage::threshold() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return threshold_;
}

bool Stage::Configure(const int32_t threshold, const StageParam *const param,
                      const std::shared_ptr<ParamManager> &param_manager) {
  OMNI_RETURN_VAL_IF(threshold <= 0 || param_manager == nullptr, false);
  std::lock_guard<std::mutex> lock(mutex_);
  threshold_ = threshold;
  param_manager_ = param_manager;
  if (param == nullptr) {
    param_.reset();
    return true;
  }
  param_ = param->Clone();
  return param_ != nullptr;
}

Status Stage::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  OMNI_RETURN_VAL_IF_LOG(threshold_ <= 0, InvalidArgumentStatus("stage threshold is invalid"), ERROR,
                    "stage threshold is invalid");
  const uint64_t generation = generation_;
  ++current_count_;
  if (current_count_ >= threshold_) {
    current_count_ = 0;
    ++generation_;
    const auto param = param_;
    lock.unlock();
    const bool is_launched = Launch(param.get());
    condition_.notify_all();
    OMNI_RETURN_VAL_IF_LOG(!is_launched, InternalStatus("stage launch failed"), ERROR,
                      "stage launch failed: " << name());
    return Status();
  }
  condition_.wait(lock, [this, generation]() { return generation_ != generation; });
  return Status();
}

} // namespace graph
} // namespace omni_runtime
