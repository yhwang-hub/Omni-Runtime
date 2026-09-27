#include "Omni-Runtime/utils/thread_pool.h"
#include "Omni-Runtime/utils/utils.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace omni_runtime::utils {
namespace {

bool ReturnFalseWhenConditionMatches() {
  OMNI_RETURN_VAL_IF(true, false);
  return true;
}

Status ReturnStatusWhenConditionMatches() {
  OMNI_RETURN_VAL_IF(true, Status::Internal("checked", SOURCE_LOCATION()));
  return Status::OK();
}

TEST(LoggerTest, ReturnsExpectedValues) {
  EXPECT_FALSE(ReturnFalseWhenConditionMatches());
  EXPECT_FALSE(ReturnStatusWhenConditionMatches().ok());
}

TEST(StatusTest, KeepsErrorInformation) {
  const Status status = Status::Internal("semantics preserved", SOURCE_LOCATION());
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kInternal);
  EXPECT_EQ(status.message(), "semantics preserved");
  EXPECT_NE(status.location().line, 0);
}

bool ReturnFalseIfNotOk() {
  OMNI_RETURN_FALSE_IF_NOT_OK(Status::NotFound("missing", SOURCE_LOCATION()));
  return true;
}

TEST(StatusMacroTest, ConvertsFailedStatusToFalse) {
  EXPECT_FALSE(ReturnFalseIfNotOk());
}

TEST(ThreadPoolTest, PostsCallableAndArguments) {
  ThreadPool pool("utils-test", 1U);
  const std::vector<int> values = {1, 2, 3};
  auto future = pool.SafePost(
      [](const std::vector<int> &input) { return input.size(); }, values);
  EXPECT_EQ(future.get(), 3U);
  pool.Wait();
}

} // namespace
} // namespace omni_runtime::utils
