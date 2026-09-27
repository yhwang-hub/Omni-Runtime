#include <memory>

#include "gtest/gtest.h"

#include "Omni-Runtime/graph/message/message_include.h"

namespace omni_runtime {
namespace graph {
namespace {

class IntegerMessage final : public MessageParam {
public:
  explicit IntegerMessage(const int32_t data) : data_(data) {
  }

  int32_t data() const {
    return data_;
  }

private:
  int32_t data_ = 0;
};

TEST(MessageTest, SupportsQueuePushStrategies) {
  MessageManager *const manager = MessageManager::Instance();
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(manager->Clear().ok());
  ASSERT_TRUE(manager->CreateTopic<IntegerMessage>("queue", 1U).ok());

  ASSERT_TRUE(manager
                  ->SendTopicValue("queue", std::make_shared<IntegerMessage>(1),
                                   MessagePushStrategy::kReplace)
                  .ok());
  ASSERT_TRUE(manager
                  ->SendTopicValue("queue", std::make_shared<IntegerMessage>(2),
                                   MessagePushStrategy::kReplace)
                  .ok());
  std::shared_ptr<IntegerMessage> output;
  ASSERT_TRUE(manager->ReceiveTopicValue("queue", 50, &output).ok());
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->data(), 2);

  ASSERT_TRUE(
      manager
          ->SendTopicValue("queue", std::make_shared<IntegerMessage>(3), MessagePushStrategy::kDrop)
          .ok());
  ASSERT_TRUE(
      manager
          ->SendTopicValue("queue", std::make_shared<IntegerMessage>(4), MessagePushStrategy::kDrop)
          .ok());
  ASSERT_TRUE(manager->ReceiveTopicValue("queue", 50, &output).ok());
  EXPECT_EQ(output->data(), 3);
  EXPECT_TRUE(manager->RemoveTopic("queue").ok());
}

TEST(MessageTest, PublishesToIndependentSubscribers) {
  MessageManager *const manager = MessageManager::Instance();
  ASSERT_NE(manager, nullptr);
  ASSERT_TRUE(manager->Clear().ok());
  const ConnectionId first = manager->BindTopic<IntegerMessage>("broadcast", 2U);
  const ConnectionId second = manager->BindTopic<IntegerMessage>("broadcast", 2U);
  ASSERT_NE(first, kInvalidConnectionId);
  ASSERT_NE(second, kInvalidConnectionId);

  ASSERT_TRUE(manager
                  ->PublishTopicValue("broadcast", std::make_shared<IntegerMessage>(7),
                                      MessagePushStrategy::kWait)
                  .ok());
  std::shared_ptr<IntegerMessage> first_output;
  std::shared_ptr<IntegerMessage> second_output;
  ASSERT_TRUE(manager->SubscribeTopicValue(first, 50, &first_output).ok());
  ASSERT_TRUE(manager->SubscribeTopicValue(second, 50, &second_output).ok());
  ASSERT_NE(first_output, nullptr);
  ASSERT_NE(second_output, nullptr);
  EXPECT_EQ(first_output->data(), 7);
  EXPECT_EQ(second_output->data(), 7);
  EXPECT_TRUE(manager->DropTopic("broadcast").ok());
}

} // namespace
} // namespace graph
} // namespace omni_runtime
