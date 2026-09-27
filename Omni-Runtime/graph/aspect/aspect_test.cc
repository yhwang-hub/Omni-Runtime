#include "gtest/gtest.h"

#include "Omni-Runtime/graph/aspect/aspect_include.h"

namespace omni_runtime {
namespace graph {
namespace {

class CountingAspect final : public Aspect {
public:
  Status BeginInit() final {
    ++begin_init_count_;
    return Status();
  }

  bool FinishInit(const Status &status) final {
    is_finish_init_ok_ = status.ok();
    ++finish_init_count_;
    return true;
  }

  Status BeginRun() final {
    ++begin_run_count_;
    return Status();
  }

  bool FinishRun(const Status &status) final {
    is_finish_run_ok_ = status.ok();
    ++finish_run_count_;
    return true;
  }

  int32_t begin_init_count() const {
    return begin_init_count_;
  }

  int32_t finish_init_count() const {
    return finish_init_count_;
  }

  int32_t begin_run_count() const {
    return begin_run_count_;
  }

  int32_t finish_run_count() const {
    return finish_run_count_;
  }

  bool is_finish_init_ok() const {
    return is_finish_init_ok_;
  }

  bool is_finish_run_ok() const {
    return is_finish_run_ok_;
  }

private:
  int32_t begin_init_count_ = 0;
  int32_t finish_init_count_ = 0;
  int32_t begin_run_count_ = 0;
  int32_t finish_run_count_ = 0;
  bool is_finish_init_ok_ = false;
  bool is_finish_run_ok_ = false;
};

TEST(AspectTest, ReflectsLifecycleInRegistrationOrder) {
  const auto param_manager = std::make_shared<ParamManager>();
  const auto event_manager = std::make_shared<EventManager>();
  AspectManager manager;
  ASSERT_TRUE(manager.SetContext(param_manager, event_manager, "test_node"));

  const auto aspect = std::make_shared<CountingAspect>();
  DefaultPassedParam param;
  ASSERT_TRUE(manager.Add(aspect, &param).ok());
  EXPECT_EQ(manager.size(), 1U);
  EXPECT_TRUE(manager.Reflect(AspectType::kBeginInit, Status()).ok());
  EXPECT_TRUE(manager.Reflect(AspectType::kFinishInit, Status()).ok());
  EXPECT_TRUE(manager.Reflect(AspectType::kBeginRun, Status()).ok());
  EXPECT_TRUE(manager.Reflect(AspectType::kFinishRun, Status()).ok());

  EXPECT_EQ(aspect->begin_init_count(), 1);
  EXPECT_EQ(aspect->finish_init_count(), 1);
  EXPECT_EQ(aspect->begin_run_count(), 1);
  EXPECT_EQ(aspect->finish_run_count(), 1);
  EXPECT_TRUE(aspect->is_finish_init_ok());
  EXPECT_TRUE(aspect->is_finish_run_ok());
  EXPECT_TRUE(manager.PopLast().ok());
  EXPECT_EQ(manager.size(), 0U);
}

} // namespace
} // namespace graph
} // namespace omni_runtime
