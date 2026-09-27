#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <type_traits>

#include "Omni-Runtime/utils/logger.h"

#include "Omni-Runtime/graph/basic/status.h"
#include "Omni-Runtime/graph/basic/types.h"
#include "Omni-Runtime/graph/message/message_define.h"
#include "Omni-Runtime/graph/param/message_param.h"

namespace omni_runtime {
namespace graph {

class MessageBase {
public:
  explicit MessageBase(const std::size_t capacity, const ConnectionId connection_id)
      : capacity_(capacity), connection_id_(connection_id) {
  }
  virtual ~MessageBase() = default;

  MessageBase(const MessageBase &) = delete;
  MessageBase &operator=(const MessageBase &) = delete;

  std::size_t capacity() const {
    return capacity_;
  }

  ConnectionId connection_id() const {
    return connection_id_;
  }

  virtual bool Clear() = 0;

private:
  std::size_t capacity_ = 0U;
  ConnectionId connection_id_ = kInvalidConnectionId;
};

template <typename T> class Message final : public MessageBase {
public:
  static_assert(std::is_base_of<MessageParam, T>::value,
                "Message payload must derive from MessageParam");

  explicit Message(const std::size_t capacity, const ConnectionId connection_id)
      : MessageBase(capacity, connection_id) {
  }

  ~Message() override {
    Clear();
  }

  Status Send(const std::shared_ptr<T> &message_param, const MessagePushStrategy strategy) {
    OMNI_RETURN_VAL_IF_LOG(message_param == nullptr, InvalidArgumentStatus("message value is null"),
                      ERROR, "message value is null");
    OMNI_RETURN_VAL_IF_LOG(capacity() == 0U, InvalidArgumentStatus("message capacity is zero"), ERROR,
                      "message capacity is zero");

    std::unique_lock<std::mutex> lock(mutex_);
    OMNI_RETURN_VAL_IF_LOG(is_stopped_, InternalStatus("message queue is stopped"), ERROR,
                      "message queue is stopped");
    if (strategy == MessagePushStrategy::kWait) {
      not_full_cv_.wait(lock, [this]() { return is_stopped_ || queue_.size() < capacity(); });
      OMNI_RETURN_VAL_IF_LOG(is_stopped_, InternalStatus("message queue is stopped"), ERROR,
                        "message queue is stopped");
    } else if (queue_.size() >= capacity()) {
      if (strategy == MessagePushStrategy::kDrop) {
        return Status();
      }
      queue_.pop_front();
    }

    queue_.emplace_back(message_param);
    lock.unlock();
    not_empty_cv_.notify_one();
    return Status();
  }

  Status Receive(const Milliseconds timeout_ms, std::shared_ptr<T> *const message_param) {
    OMNI_RETURN_VAL_IF_LOG(message_param == nullptr, InvalidArgumentStatus("message output is null"),
                      ERROR, "message output is null");
    OMNI_RETURN_VAL_IF_LOG(timeout_ms < 0, InvalidArgumentStatus("timeout is negative"), ERROR,
                      "timeout is negative");

    std::unique_lock<std::mutex> lock(mutex_);
    const bool is_ready =
        not_empty_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                               [this]() { return is_stopped_ || !queue_.empty(); });
    OMNI_RETURN_VAL_IF_LOG(!is_ready || queue_.empty(),
                      Status::Cancelled("message receive timeout", SOURCE_LOCATION()), WARN,
                      "message receive timeout");
    *message_param = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
    not_full_cv_.notify_one();
    return Status();
  }

  bool Clear() final {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.clear();
      is_stopped_ = true;
    }
    not_empty_cv_.notify_all();
    not_full_cv_.notify_all();
    return true;
  }

private:
  std::mutex mutex_;
  std::condition_variable not_empty_cv_;
  std::condition_variable not_full_cv_;
  std::deque<std::shared_ptr<T>> queue_;
  bool is_stopped_ = false;
};

} // namespace graph
} // namespace omni_runtime
