#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <tuple>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace omni_runtime::utils {

class ThreadPool final {
public:
  explicit ThreadPool(const std::string &name, const std::size_t thread_count);
  ~ThreadPool();

  template <typename Function, typename... Arguments>
  auto SafePost(Function &&function, Arguments &&...arguments)
      -> std::future<std::invoke_result_t<Function, Arguments...>> {
    using Result = std::invoke_result_t<Function, Arguments...>;
    auto invocation = [function = std::forward<Function>(function),
                       arguments = std::make_tuple(std::forward<Arguments>(arguments)...)]() mutable -> Result {
      return std::apply(
          [&function](auto &...argument) { return std::invoke(function, argument...); }, arguments);
    };
    auto task = std::make_shared<std::packaged_task<Result()>>(std::move(invocation));
    std::future<Result> future = task->get_future();
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return future;
  }

  void Wait();
  void Stop();
  static bool IsWorkerThread();

private:
  void Start();

  std::string name_;
  std::size_t thread_count_;
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable completed_;
  std::size_t active_count_ = 0U;
  bool is_stopping_ = false;
  static thread_local bool is_worker_thread_;
};

inline thread_local bool ThreadPool::is_worker_thread_ = false;

} // namespace omni_runtime::utils
