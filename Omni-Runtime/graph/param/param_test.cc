#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "Omni-Runtime/graph/param/param_include.h"

namespace omni_runtime {
namespace graph {
namespace {

class CountingParam final : public Param {
public:
  Status Init() final {
    ++init_count_;
    return Status();
  }

  Status Setup() final {
    ++setup_count_;
    return Status();
  }

  bool Reset(const Status &status) final {
    is_last_status_ok_ = status.ok();
    ++reset_count_;
    return true;
  }

  Status Destroy() final {
    ++destroy_count_;
    return Status();
  }

  int32_t init_count() const {
    return init_count_;
  }

  int32_t setup_count() const {
    return setup_count_;
  }

  int32_t reset_count() const {
    return reset_count_;
  }

  int32_t destroy_count() const {
    return destroy_count_;
  }

  bool is_last_status_ok() const {
    return is_last_status_ok_;
  }

private:
  int32_t init_count_ = 0;
  int32_t setup_count_ = 0;
  int32_t reset_count_ = 0;
  int32_t destroy_count_ = 0;
  bool is_last_status_ok_ = false;
};

class OtherParam final : public Param {};

TEST(ParamTest, ManagesTypedParamsLifecycleAndBacktrace) {
  ParamManager manager;
  ASSERT_TRUE(manager.Create<CountingParam>("state", true).ok());
  EXPECT_TRUE(manager.Create<CountingParam>("state", true).ok());
  EXPECT_FALSE(manager.Create<OtherParam>("state", false).ok());
  EXPECT_TRUE(manager.Has("state"));

  const std::shared_ptr<CountingParam> param = manager.Get<CountingParam>("state");
  ASSERT_NE(param, nullptr);
  EXPECT_EQ(param->key(), "state");
  EXPECT_TRUE(param->AddBacktrace("node_a").ok());
  EXPECT_TRUE(param->AddBacktrace("node_b").ok());
  std::vector<std::string> traces;
  ASSERT_TRUE(param->GetBacktrace(&traces).ok());
  EXPECT_EQ(traces, (std::vector<std::string>{"node_a", "node_b"}));

  EXPECT_TRUE(manager.Init().ok());
  EXPECT_TRUE(manager.Setup().ok());
  EXPECT_TRUE(manager.Reset(Status()));
  EXPECT_TRUE(manager.Destroy().ok());
  EXPECT_EQ(param->init_count(), 1);
  EXPECT_EQ(param->setup_count(), 1);
  EXPECT_EQ(param->reset_count(), 1);
  EXPECT_EQ(param->destroy_count(), 1);
  EXPECT_TRUE(param->is_last_status_ok());

  EXPECT_TRUE(manager.Remove("state").ok());
  EXPECT_FALSE(manager.Has("state"));
  EXPECT_EQ(manager.Get<CountingParam>("state"), nullptr);
}

} // namespace
} // namespace graph
} // namespace omni_runtime
