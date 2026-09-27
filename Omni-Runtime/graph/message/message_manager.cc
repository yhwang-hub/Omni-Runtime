#include "Omni-Runtime/graph/message/message_manager.h"

namespace omni_runtime {
namespace graph {

MessageManager::~MessageManager() {
  static_cast<void>(Clear());
}

MessageManager *MessageManager::Instance() {
  static MessageManager manager;
  return &manager;
}

Status MessageManager::RemoveTopic(const std::string &topic) {
  std::shared_ptr<MessageBase> message;
  {
    std::lock_guard<std::mutex> lock(send_receive_mutex_);
    const auto iter = send_receive_topics_.find(topic);
    OMNI_RETURN_VAL_IF_LOG(iter == send_receive_topics_.end(),
                      NotFoundStatus("topic not found: " + topic), ERROR,
                      "topic not found: " << topic);
    message = iter->second;
    send_receive_topics_.erase(iter);
  }
  message->Clear();
  return Status();
}

Status MessageManager::DetachConnection(const std::string &topic,
                                        const ConnectionId connection_id) {
  std::shared_ptr<MessageBase> message;
  {
    std::lock_guard<std::mutex> lock(publish_subscribe_mutex_);
    const auto topic_iter = subscribers_.find(topic);
    OMNI_RETURN_VAL_IF_LOG(topic_iter == subscribers_.end(), NotFoundStatus("topic not found: " + topic),
                      ERROR, "topic not found: " << topic);
    const auto connection_iter = topic_iter->second.find(connection_id);
    OMNI_RETURN_VAL_IF_LOG(connection_iter == topic_iter->second.end(),
                      NotFoundStatus("connection not found"), ERROR,
                      "connection not found: " << connection_id);
    message = connection_iter->second;
    topic_iter->second.erase(connection_iter);
    connections_.erase(connection_id);
    if (topic_iter->second.empty()) {
      subscribers_.erase(topic_iter);
    }
  }
  message->Clear();
  return Status();
}

Status MessageManager::DropTopic(const std::string &topic) {
  SubscriberMap dropped;
  {
    std::lock_guard<std::mutex> lock(publish_subscribe_mutex_);
    const auto iter = subscribers_.find(topic);
    OMNI_RETURN_VAL_IF_LOG(iter == subscribers_.end(), NotFoundStatus("topic not found: " + topic),
                      ERROR, "topic not found: " << topic);
    dropped = std::move(iter->second);
    subscribers_.erase(iter);
    for (const auto &item : dropped) {
      connections_.erase(item.first);
    }
  }
  for (const auto &item : dropped) {
    item.second->Clear();
  }
  return Status();
}

Status MessageManager::Clear() {
  std::vector<std::shared_ptr<MessageBase>> messages;
  {
    std::lock_guard<std::mutex> lock(send_receive_mutex_);
    for (const auto &item : send_receive_topics_) {
      messages.emplace_back(item.second);
    }
    send_receive_topics_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(publish_subscribe_mutex_);
    for (const auto &item : connections_) {
      messages.emplace_back(item.second);
    }
    subscribers_.clear();
    connections_.clear();
    last_connection_id_ = 0;
  }
  for (const auto &message : messages) {
    message->Clear();
  }
  return Status();
}

} // namespace graph
} // namespace omni_runtime
