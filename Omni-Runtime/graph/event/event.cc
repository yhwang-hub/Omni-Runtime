#include "Omni-Runtime/graph/event/event.h"

namespace omni_runtime {
namespace graph {

Status Event::Process(const EventType type, const EventAsyncStrategy strategy) {
  if (type == EventType::kSync) {
    OMNI_RETURN_VAL_IF_LOG(!Trigger(param_.get()), InternalStatus("event trigger failed"), ERROR,
                      "event trigger failed: " << name());
    return Status();
  }
  OMNI_RETURN_VAL_IF_LOG(type != EventType::kAsync, InvalidArgumentStatus("unknown event type"), ERROR,
                    "unknown event type");
  OMNI_RETURN_VAL_IF_LOG(thread_pool_ == nullptr, InternalStatus("thread pool is unavailable"), ERROR,
                    "thread pool is unavailable");
  const std::shared_future<void> future = AsyncProcess(strategy);
  OMNI_RETURN_VAL_IF_LOG(!future.valid(), InternalStatus("event future is invalid"), ERROR,
                    "event future is invalid");
  return Status();
}

std::shared_future<void> Event::AsyncProcess(const EventAsyncStrategy strategy) {
  OMNI_RETURN_VAL_IF_LOG(thread_pool_ == nullptr, std::shared_future<void>(), ERROR,
                    "thread pool is unavailable");
  std::shared_future<void> future =
      thread_pool_
          ->SafePost([this]() {
            OMNI_RETURN_IF_LOG(!Trigger(param_.get()), ERROR, "event trigger failed: " << name());
          })
          .share();

  if (strategy == EventAsyncStrategy::kPipelineRunFinish) {
    std::lock_guard<std::mutex> lock(run_futures_mutex_);
    run_futures_.emplace_back(future);
  } else if (strategy == EventAsyncStrategy::kPipelineDestroy) {
    std::lock_guard<std::mutex> lock(destroy_futures_mutex_);
    destroy_futures_.emplace_back(future);
  }
  return future;
}

bool Event::Wait(const EventAsyncStrategy strategy) {
  std::vector<std::shared_future<void>> futures;
  if (strategy == EventAsyncStrategy::kPipelineRunFinish) {
    std::lock_guard<std::mutex> lock(run_futures_mutex_);
    futures.swap(run_futures_);
  } else if (strategy == EventAsyncStrategy::kPipelineDestroy) {
    std::lock_guard<std::mutex> lock(destroy_futures_mutex_);
    futures.swap(destroy_futures_);
  } else {
    return false;
  }
  for (const auto &future : futures) {
    if (future.valid()) {
      future.wait();
    }
  }
  return true;
}

bool Event::SetContext(const std::shared_ptr<ParamManager> &param_manager,
                       const std::shared_ptr<omni_runtime::utils::ThreadPool> &thread_pool) {
  OMNI_RETURN_VAL_IF(param_manager == nullptr || thread_pool == nullptr, false);
  param_manager_ = param_manager;
  thread_pool_ = thread_pool;
  return true;
}

bool Event::SetParam(const EventParam *const param) {
  if (param == nullptr) {
    param_.reset();
    return true;
  }
  param_ = param->Clone();
  return param_ != nullptr;
}

} // namespace graph
} // namespace omni_runtime
