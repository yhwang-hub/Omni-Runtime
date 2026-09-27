#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "Omni-Runtime/graph/element/element_include.h"
#include "Omni-Runtime/graph/pipeline/pipeline.h"

namespace omni_runtime {
namespace graph {
namespace {

class RecordingNode final : public Node {
public:
  explicit RecordingNode(const std::string &tag, std::vector<std::string> *const records,
                         std::mutex *const mutex)
      : tag_(tag), records_(records), mutex_(mutex) {
  }

  int32_t run_count() const {
    return run_count_;
  }

protected:
  Status Run() final {
    OMNI_RETURN_VAL_IF_LOG(records_ == nullptr || mutex_ == nullptr,
                      InvalidArgumentStatus("record storage is null"), ERROR,
                      "record storage is null");
    std::lock_guard<std::mutex> lock(*mutex_);
    records_->emplace_back(tag_);
    ++run_count_;
    return Status();
  }

private:
  std::string tag_;
  std::vector<std::string> *records_ = nullptr;
  std::mutex *mutex_ = nullptr;
  int32_t run_count_ = 0;
};

NodeInfo MakeNodeInfo(const std::string &name,
                      const std::vector<std::shared_ptr<Element>> &dependencies) {
  NodeInfo info;
  info.name = name;
  info.loop_count = 1U;
  info.dependencies = dependencies;
  return info;
}

TEST(ElementTest, RunsDagInTopologicalBatches) {
  Pipeline pipeline;
  ASSERT_TRUE(pipeline.set_thread_count(4U));
  std::vector<std::string> records;
  std::mutex mutex;
  const auto first =
      pipeline.RegisterNode<RecordingNode>(MakeNodeInfo("first", {}), "first", &records, &mutex);
  const auto second =
      pipeline.RegisterNode<RecordingNode>(MakeNodeInfo("second", {}), "second", &records, &mutex);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  const auto tail = pipeline.RegisterNode<RecordingNode>(MakeNodeInfo("tail", {first, second}),
                                                         "tail", &records, &mutex);
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(pipeline.MaxParallelism(), 2U);

  ASSERT_TRUE(pipeline.Process(1U).ok());
  ASSERT_EQ(records.size(), 3U);
  const auto first_iter = std::find(records.begin(), records.end(), "first");
  const auto second_iter = std::find(records.begin(), records.end(), "second");
  const auto tail_iter = std::find(records.begin(), records.end(), "tail");
  ASSERT_NE(first_iter, records.end());
  ASSERT_NE(second_iter, records.end());
  ASSERT_NE(tail_iter, records.end());
  EXPECT_LT(first_iter, tail_iter);
  EXPECT_LT(second_iter, tail_iter);
  EXPECT_EQ(tail->run_count(), 1);
}

TEST(ElementTest, RejectsCyclesAndRunsClusterChildren) {
  std::vector<std::string> records;
  std::mutex mutex;
  const auto first = std::make_shared<RecordingNode>("first", &records, &mutex);
  const auto second = std::make_shared<RecordingNode>("second", &records, &mutex);
  ASSERT_TRUE(second->AddDependency(first).ok());
  EXPECT_FALSE(first->AddDependency(second).ok());

  Pipeline pipeline;
  ASSERT_TRUE(pipeline.set_thread_count(2U));
  const auto child_a =
      pipeline.CreateNode<RecordingNode>(MakeNodeInfo("child_a", {}), "child_a", &records, &mutex);
  const auto child_b =
      pipeline.CreateNode<RecordingNode>(MakeNodeInfo("child_b", {}), "child_b", &records, &mutex);
  ASSERT_NE(child_a, nullptr);
  ASSERT_NE(child_b, nullptr);
  ASSERT_TRUE(child_b->AddDependency(child_a).ok());
  const auto cluster = pipeline.CreateGroup<Cluster>({child_a, child_b}, {}, "cluster", 1U);
  ASSERT_NE(cluster, nullptr);
  ASSERT_TRUE(pipeline.RegisterElement(cluster, {}, "cluster", 1U).ok());
  ASSERT_TRUE(pipeline.Process(1U).ok());
  EXPECT_EQ(child_a->run_count(), 1);
  EXPECT_EQ(child_b->run_count(), 1);
}

} // namespace
} // namespace graph
} // namespace omni_runtime
