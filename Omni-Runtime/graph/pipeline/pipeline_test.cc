#include <atomic>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>

#include "gtest/gtest.h"

#include "Omni-Runtime/graph/pipeline/pipeline_include.h"

namespace omni_runtime {
namespace graph {
namespace {

class LifecycleParam final : public Param {
public:
  Status Setup() final {
    ++setup_count_;
    return Status();
  }

  bool Reset(const Status &status) final {
    is_last_status_ok_ = status.ok();
    ++reset_count_;
    return true;
  }

  int32_t setup_count() const {
    return setup_count_;
  }

  int32_t reset_count() const {
    return reset_count_;
  }

  bool is_last_status_ok() const {
    return is_last_status_ok_;
  }

private:
  int32_t setup_count_ = 0;
  int32_t reset_count_ = 0;
  bool is_last_status_ok_ = false;
};

class CounterNode final : public Node {
public:
  explicit CounterNode(std::atomic<int32_t> *const run_count) : run_count_(run_count) {
  }

protected:
  Status Run() final {
    OMNI_RETURN_VAL_IF_LOG(run_count_ == nullptr, InvalidArgumentStatus("run count is null"), ERROR,
                      "run count is null");
    run_count_->fetch_add(1, std::memory_order_release);
    return Status();
  }

private:
  std::atomic<int32_t> *run_count_ = nullptr;
};

class StorageNode final : public Node {
public:
  static void ResetRunCount() {
    run_count_.store(0, std::memory_order_release);
  }

  static int32_t run_count() {
    return run_count_.load(std::memory_order_acquire);
  }

protected:
  Status Run() final {
    run_count_.fetch_add(1, std::memory_order_release);
    return Status();
  }

private:
  inline static std::atomic<int32_t> run_count_ = 0;
};

class StorageAspect final : public Aspect {
public:
  static void ResetBeginRunCount() {
    begin_run_count_.store(0, std::memory_order_release);
  }

  static int32_t begin_run_count() {
    return begin_run_count_.load(std::memory_order_acquire);
  }

  Status BeginRun() final {
    begin_run_count_.fetch_add(1, std::memory_order_release);
    return Status();
  }

private:
  inline static std::atomic<int32_t> begin_run_count_ = 0;
};

class StorageEvent final : public Event {
protected:
  bool Trigger(const EventParam *const param) final {
    static_cast<void>(param);
    return true;
  }
};

class StorageDaemon final : public Daemon {
protected:
  bool DaemonTask(const DaemonParam *const param) final {
    static_cast<void>(param);
    return true;
  }
};

TEST(PipelineTest, ProcessesNodesAndParamLifecycle) {
  const std::shared_ptr<Pipeline> pipeline = PipelineFactory::Create();
  ASSERT_NE(pipeline, nullptr);
  ASSERT_TRUE(pipeline->set_name("pipeline_test"));
  ASSERT_TRUE(pipeline->set_thread_count(2U));
  ASSERT_TRUE(pipeline->CreateParam<LifecycleParam>("state", true).ok());
  const std::shared_ptr<LifecycleParam> param = pipeline->GetParam<LifecycleParam>("state");
  ASSERT_NE(param, nullptr);

  std::atomic<int32_t> run_count = 0;
  NodeInfo info;
  info.name = "counter";
  info.loop_count = 1U;
  const auto node = pipeline->RegisterNode<CounterNode>(info, &run_count);
  ASSERT_NE(node, nullptr);
  std::ostringstream graph_stream;
  ASSERT_TRUE(pipeline->Dump(&graph_stream).ok());
  EXPECT_NE(graph_stream.str().find("counter"), std::string::npos);

  ASSERT_TRUE(pipeline->Process(2U).ok());
  EXPECT_EQ(run_count.load(std::memory_order_acquire), 2);
  EXPECT_EQ(param->setup_count(), 2);
  EXPECT_EQ(param->reset_count(), 2);
  EXPECT_TRUE(param->is_last_status_ok());
  std::ostringstream perf_stream;
  ASSERT_TRUE(pipeline->Perf(&perf_stream).ok());
  EXPECT_NE(perf_stream.str().find("counter\t2"), std::string::npos);
  EXPECT_TRUE(PipelineFactory::Remove(pipeline).ok());
}

TEST(PipelineTest, PoolsInitializedPipelines) {
  PipelineManager manager;
  const auto pipeline = std::make_shared<Pipeline>();
  ASSERT_TRUE(pipeline->set_thread_count(1U));
  ASSERT_TRUE(manager.Add(pipeline).ok());
  EXPECT_TRUE(manager.Find(pipeline));
  EXPECT_EQ(manager.size(), 1U);
  ASSERT_TRUE(manager.Init().ok());
  EXPECT_TRUE(manager.Run().ok());
  EXPECT_TRUE(manager.Destroy().ok());
  EXPECT_TRUE(manager.Clear().ok());
  EXPECT_EQ(manager.size(), 0U);
}

TEST(PipelineTest, SavesAndLoadsRegisteredElementTypes) {
  ASSERT_TRUE(StorageFactory::Register<StorageNode>());
  ASSERT_TRUE(StorageFactory::Register<StorageAspect>());
  ASSERT_TRUE(StorageFactory::Register<StorageEvent>());
  ASSERT_TRUE(StorageFactory::Register<StorageDaemon>());
  ASSERT_TRUE(StorageFactory::Register<LifecycleParam>());
  StorageNode::ResetRunCount();
  StorageAspect::ResetBeginRunCount();
  Pipeline source;
  ASSERT_TRUE(source.set_thread_count(2U));
  ASSERT_TRUE(source.CreateParam<LifecycleParam>("stored_param", true).ok());
  NodeInfo root_info;
  root_info.name = "root";
  const auto root = source.RegisterNode<StorageNode>(root_info);
  ASSERT_NE(root, nullptr);
  NodeInfo tail_info;
  tail_info.name = "tail";
  tail_info.dependencies = {root};
  ASSERT_NE(source.RegisterNode<StorageNode>(tail_info), nullptr);
  DefaultPassedParam passed_param;
  ASSERT_TRUE(source.AddAspect<StorageAspect>({}, &passed_param).ok());
  ASSERT_TRUE(
      (source.AddEvent<StorageEvent, DefaultPassedParam>("stored_event", &passed_param).ok()));
  ASSERT_TRUE((source.AddDaemon<StorageDaemon, DefaultPassedParam>(1000, &passed_param).ok()));
  ASSERT_TRUE((source.AddStage<Stage, DefaultPassedParam>("stored_stage", 1, &passed_param).ok()));

  const std::filesystem::path storage_path =
      std::filesystem::temp_directory_path() / "zomni_graph_pipeline_test.bin";
  ASSERT_TRUE(source.Save(storage_path.string()).ok());
  Pipeline restored;
  ASSERT_TRUE(restored.Load(storage_path.string()).ok());
  ASSERT_NE(restored.GetParam<LifecycleParam>("stored_param"), nullptr);
  ASSERT_TRUE(restored.Process(1U).ok());
  EXPECT_EQ(StorageNode::run_count(), 2);
  EXPECT_EQ(StorageAspect::begin_run_count(), 2);
  EXPECT_EQ(std::remove(storage_path.string().c_str()), 0);
}

} // namespace
} // namespace graph
} // namespace omni_runtime
