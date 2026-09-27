#include <atomic>
#include <future>
#include <memory>

#include "gtest/gtest.h"

#include "Omni-Runtime/graph/stage/stage_include.h"

namespace omni_runtime {
namespace graph {
namespace {

class CountingStage final : public Stage {
public:
  static void ResetCount() {
    launch_count_.store(0, std::memory_order_release);
  }

  static int32_t launch_count() {
    return launch_count_.load(std::memory_order_acquire);
  }

protected:
  bool Launch(const StageParam *const param) final {
    static_cast<void>(param);
    launch_count_.fetch_add(1, std::memory_order_release);
    return true;
  }

private:
  inline static std::atomic<int32_t> launch_count_ = 0;
};

TEST(StageTest, ReleasesAllWaitersAtThreshold) {
  CountingStage::ResetCount();
  const auto param_manager = std::make_shared<ParamManager>();
  StageManager manager;
  ASSERT_TRUE(manager.SetParamManager(param_manager));
  DefaultPassedParam param;
  ASSERT_TRUE((manager.Create<CountingStage, DefaultPassedParam>("barrier", 2, &param).ok()));
  ASSERT_TRUE(manager.Init().ok());

  auto first =
      std::async(std::launch::async, [&manager]() { return manager.WaitForReady("barrier"); });
  auto second =
      std::async(std::launch::async, [&manager]() { return manager.WaitForReady("barrier"); });
  EXPECT_TRUE(first.get().ok());
  EXPECT_TRUE(second.get().ok());
  EXPECT_EQ(CountingStage::launch_count(), 1);
  EXPECT_TRUE(manager.Destroy().ok());
}

} // namespace
} // namespace graph
} // namespace omni_runtime
