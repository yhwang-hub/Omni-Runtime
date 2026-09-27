#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/utils/logger.h"

#include "Omni-Runtime/utils/status_macros.h"
#include "Omni-Runtime/graph/message/message.h"

namespace omni_runtime {
namespace graph {

class MessageManager final {
public:
  MessageManager() = default;
  ~MessageManager();

  MessageManager(const MessageManager &) = delete;
  MessageManager &operator=(const MessageManager &) = delete;

  static MessageManager *Instance();

  template <typename T> Status CreateTopic(const std::string &topic, const std::size_t capacity) {
    static_assert(std::is_base_of<MessageParam, T>::value,
                  "Message payload must derive from MessageParam");
    OMNI_RETURN_VAL_IF_LOG(topic.empty(), InvalidArgumentStatus("topic is empty"), ERROR,
                      "topic is empty");
    OMNI_RETURN_VAL_IF_LOG(capacity == 0U, InvalidArgumentStatus("capacity is zero"), ERROR,
                      "capacity is zero");
    std::lock_guard<std::mutex> lock(send_receive_mutex_);
    const auto iter = send_receive_topics_.find(topic);
    if (iter != send_receive_topics_.end()) {
      const auto message = std::dynamic_pointer_cast<Message<T>>(iter->second);
      OMNI_RETURN_VAL_IF_LOG(message == nullptr || message->capacity() != capacity,
                        InvalidArgumentStatus("topic conflicts: " + topic), ERROR,
                        "topic conflicts: " << topic);
      return Status();
    }
    send_receive_topics_.emplace(topic,
                                 std::make_shared<Message<T>>(capacity, kInvalidConnectionId));
    return Status();
  }

  Status RemoveTopic(const std::string &topic);

  template <typename T>
  Status SendTopicValue(const std::string &topic, const std::shared_ptr<T> &message_param,
                        const MessagePushStrategy strategy) {
    std::shared_ptr<MessageBase> base;
    {
      std::lock_guard<std::mutex> lock(send_receive_mutex_);
      const auto iter = send_receive_topics_.find(topic);
      OMNI_RETURN_VAL_IF_LOG(iter == send_receive_topics_.end(),
                        NotFoundStatus("topic not found: " + topic), ERROR,
                        "topic not found: " << topic);
      base = iter->second;
    }
    const auto message = std::dynamic_pointer_cast<Message<T>>(base);
    OMNI_RETURN_VAL_IF_LOG(message == nullptr, InvalidArgumentStatus("topic type mismatch"), ERROR,
                      "topic type mismatch: " << topic);
    return message->Send(message_param, strategy);
  }

  template <typename T>
  Status ReceiveTopicValue(const std::string &topic, const Milliseconds timeout_ms,
                           std::shared_ptr<T> *const message_param) {
    OMNI_RETURN_VAL_IF_LOG(message_param == nullptr, InvalidArgumentStatus("message output is null"),
                      ERROR, "message output is null");
    std::shared_ptr<MessageBase> base;
    {
      std::lock_guard<std::mutex> lock(send_receive_mutex_);
      const auto iter = send_receive_topics_.find(topic);
      OMNI_RETURN_VAL_IF_LOG(iter == send_receive_topics_.end(),
                        NotFoundStatus("topic not found: " + topic), ERROR,
                        "topic not found: " << topic);
      base = iter->second;
    }
    const auto message = std::dynamic_pointer_cast<Message<T>>(base);
    OMNI_RETURN_VAL_IF_LOG(message == nullptr, InvalidArgumentStatus("topic type mismatch"), ERROR,
                      "topic type mismatch: " << topic);
    return message->Receive(timeout_ms, message_param);
  }

  template <typename T>
  ConnectionId BindTopic(const std::string &topic, const std::size_t capacity) {
    static_assert(std::is_base_of<MessageParam, T>::value,
                  "Message payload must derive from MessageParam");
    OMNI_RETURN_VAL_IF_LOG(topic.empty() || capacity == 0U, kInvalidConnectionId, ERROR,
                      "invalid topic or capacity");
    std::lock_guard<std::mutex> lock(publish_subscribe_mutex_);
    const ConnectionId connection_id = ++last_connection_id_;
    auto message = std::make_shared<Message<T>>(capacity, connection_id);
    subscribers_[topic].emplace(connection_id, message);
    connections_.emplace(connection_id, std::move(message));
    return connection_id;
  }

  Status DetachConnection(const std::string &topic, const ConnectionId connection_id);

  template <typename T>
  Status PublishTopicValue(const std::string &topic, const std::shared_ptr<T> &message_param,
                           const MessagePushStrategy strategy) {
    std::vector<std::shared_ptr<MessageBase>> subscribers;
    {
      std::lock_guard<std::mutex> lock(publish_subscribe_mutex_);
      const auto iter = subscribers_.find(topic);
      OMNI_RETURN_VAL_IF_LOG(iter == subscribers_.end(), NotFoundStatus("topic not found: " + topic),
                        ERROR, "topic not found: " << topic);
      for (const auto &item : iter->second) {
        subscribers.emplace_back(item.second);
      }
    }
    for (const auto &base : subscribers) {
      const auto message = std::dynamic_pointer_cast<Message<T>>(base);
      OMNI_RETURN_VAL_IF_LOG(message == nullptr, InvalidArgumentStatus("topic type mismatch"), ERROR,
                        "topic type mismatch: " << topic);
      OMNI_RETURN_STATUS_IF_NOT_OK(message->Send(message_param, strategy));
    }
    return Status();
  }

  template <typename T>
  Status SubscribeTopicValue(const ConnectionId connection_id, const Milliseconds timeout_ms,
                             std::shared_ptr<T> *const message_param) {
    OMNI_RETURN_VAL_IF_LOG(message_param == nullptr, InvalidArgumentStatus("message output is null"),
                      ERROR, "message output is null");
    std::shared_ptr<MessageBase> base;
    {
      std::lock_guard<std::mutex> lock(publish_subscribe_mutex_);
      const auto iter = connections_.find(connection_id);
      OMNI_RETURN_VAL_IF_LOG(iter == connections_.end(), NotFoundStatus("connection not found"), ERROR,
                        "connection not found: " << connection_id);
      base = iter->second;
    }
    const auto message = std::dynamic_pointer_cast<Message<T>>(base);
    OMNI_RETURN_VAL_IF_LOG(message == nullptr, InvalidArgumentStatus("connection type mismatch"), ERROR,
                      "connection type mismatch: " << connection_id);
    return message->Receive(timeout_ms, message_param);
  }

  Status DropTopic(const std::string &topic);
  Status Clear();

private:
  using SubscriberMap = std::unordered_map<ConnectionId, std::shared_ptr<MessageBase>>;

  std::mutex send_receive_mutex_;
  std::unordered_map<std::string, std::shared_ptr<MessageBase>> send_receive_topics_;

  std::mutex publish_subscribe_mutex_;
  std::unordered_map<std::string, SubscriberMap> subscribers_;
  std::unordered_map<ConnectionId, std::shared_ptr<MessageBase>> connections_;
  ConnectionId last_connection_id_ = 0;
};

} // namespace graph
} // namespace omni_runtime
