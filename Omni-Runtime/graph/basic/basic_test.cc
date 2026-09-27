#include "Omni-Runtime/graph/basic/basic.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"

namespace omni_runtime {
namespace graph {
namespace {

class TestObject final : public Object {
public:
  Status Init() final {
    is_initialized_ = true;
    return Status();
  }

  Status Run() final {
    ++run_count_;
    return Status();
  }

  Status Destroy() final {
    is_initialized_ = false;
    return Status();
  }

  bool is_initialized() const {
    return is_initialized_;
  }

  int32_t run_count() const {
    return run_count_;
  }

private:
  bool is_initialized_ = false;
  int32_t run_count_ = 0;
};

class TestDescriptor final : public Descriptor {
public:
  TestDescriptor() : Descriptor("test") {
  }
};

TEST(BasicTest, AllocatesObjectAndRunsLifecycle) {
  const std::shared_ptr<TestObject> object = Allocator::MakeObject<TestObject>();
  ASSERT_NE(object, nullptr);
  EXPECT_TRUE(object->Init().ok());
  EXPECT_TRUE(object->is_initialized());
  EXPECT_TRUE(object->Run().ok());
  EXPECT_EQ(object->run_count(), 1);
  EXPECT_TRUE(object->Destroy().ok());
  EXPECT_FALSE(object->is_initialized());
}

TEST(BasicTest, DescriptorUsesStableUniqueSessions) {
  TestDescriptor first;
  TestDescriptor second;
  EXPECT_NE(first.session(), second.session());
  EXPECT_EQ(first.name(), first.session());
  EXPECT_TRUE(first.set_name("renamed"));
  EXPECT_TRUE(first.set_description("basic descriptor"));
  EXPECT_EQ(first.name(), "renamed");
  EXPECT_EQ(first.description(), "basic descriptor");
  EXPECT_FALSE(InvalidArgumentStatus("invalid").ok());
}

} // namespace
} // namespace graph
} // namespace omni_runtime
