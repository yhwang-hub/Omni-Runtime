#include "Omni-Runtime/utils/thread_pool.h"

namespace omni_runtime::utils {

ThreadPool::ThreadPool(const std::string &name, const std::size_t thread_count)
    : name_(name), thread_count_(thread_count) {
  Start();
}

ThreadPool::~ThreadPool() {
  Stop();
}

void ThreadPool::Wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  completed_.wait(lock, [this]() { return tasks_.empty() && active_count_ == 0U; });
}

void ThreadPool::Stop() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    is_stopping_ = true;
  }
  condition_.notify_all();
  for (std::thread &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();
}

bool ThreadPool::IsWorkerThread() {
  return is_worker_thread_;
}

void ThreadPool::Start() {
  for (std::size_t index = 0U; index < thread_count_; ++index) {
    workers_.emplace_back([this]() {
      is_worker_thread_ = true;
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          condition_.wait(lock, [this]() { return is_stopping_ || !tasks_.empty(); });
          if (is_stopping_ && tasks_.empty()) {
            return;
          }
          task = std::move(tasks_.front());
          tasks_.pop();
          ++active_count_;
        }
        task();
        {
          const std::lock_guard<std::mutex> lock(mutex_);
          --active_count_;
          if (tasks_.empty() && active_count_ == 0U) {
            completed_.notify_all();
          }
        }
      }
    });
  }
}

} // namespace omni_runtime::utils
