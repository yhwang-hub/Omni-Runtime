#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "gtest/gtest.h"

#include "Omni-Runtime/graph/daemon/daemon_include.h"

namespace omni_runtime {
namespace graph {
namespace {

class CountingDaemon final : public Daemon {
public:
  int32_t run_count() const {
    return run_count_.load(std::memory_order_acquire);
  }

protected:
  bool DaemonTask(const DaemonParam *const param) final {
    static_cast<void>(param);
    run_count_.fetch_add(1, std::memory_order_release);
    return true;
  }

private:
  std::atomic<int32_t> run_count_ = 0;
};

TEST(DaemonTest, RunsPeriodicallyAndStopsCleanly) {
  const auto param_manager = std::make_shared<ParamManager>();
  const auto event_manager = std::make_shared<EventManager>();
  DaemonManager manager;
  ASSERT_TRUE(manager.SetContext(param_manager, event_manager));

  const auto daemon = std::make_shared<CountingDaemon>();
  ASSERT_TRUE(daemon->set_interval_ms(5));
  DefaultPassedParam param;
  ASSERT_TRUE(daemon->SetParam(&param));
  ASSERT_TRUE(manager.Add(daemon).ok());
  ASSERT_TRUE(manager.Init().ok());
  std::this_thread::sleep_for(std::chrono::milliseconds(40));
  ASSERT_TRUE(manager.Destroy().ok());
  EXPECT_GE(daemon->run_count(), 1);
  const int32_t stopped_count = daemon->run_count();
  std::this_thread::sleep_for(std::chrono::milliseconds(15));
  EXPECT_EQ(daemon->run_count(), stopped_count);
}

} // namespace
} // namespace graph
} // namespace omni_runtime
