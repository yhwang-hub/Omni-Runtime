#include "Omni-Runtime/inference/common/text_stream/text_stream.h"

#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

namespace omni_runtime::inference {

TEST(TextStreamTest, BuffersSplitUtf8AndEmitsOnlyNewText) {
  std::vector<std::string> chunks;
  TextStream stream([&chunks](const std::string &chunk) {
    chunks.emplace_back(chunk);
    return true;
  });
  ASSERT_TRUE(stream.Append("A\xE4"));
  ASSERT_TRUE(stream.Append("\xB8"));
  EXPECT_EQ(chunks, (std::vector<std::string>{"A"}));
  ASSERT_TRUE(stream.Append("\xAD\xF0\x9F"));
  ASSERT_TRUE(stream.Append("\x98\x80"));
  ASSERT_TRUE(stream.Finish());
  EXPECT_EQ(chunks, (std::vector<std::string>{"A", "\xE4\xB8\xAD", "\xF0\x9F\x98\x80"}));
  EXPECT_EQ(stream.text(), "A\xE4\xB8\xAD\xF0\x9F\x98\x80");
}

TEST(TextStreamTest, ReplacesIncompleteAndInvalidUtf8) {
  TextStream stream(TextStreamCallback{});
  ASSERT_TRUE(stream.Append("\xC0\xAF\xED\xA0\x80\xE4\xB8"));
  ASSERT_TRUE(stream.Finish());
  EXPECT_EQ(stream.text(),
            "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD");
}

TEST(TextStreamTest, CancellationDoesNotDeliverAdditionalChunks) {
  int32_t calls = 0;
  TextStream stream([&calls](const std::string &) {
    ++calls;
    return false;
  });
  ASSERT_TRUE(stream.Append("first"));
  EXPECT_TRUE(stream.is_cancelled());
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(stream.Append("second"));
  const std::string error_log = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(error_log, "[2] Cannot append to a cancelled text stream\n");
  ASSERT_TRUE(stream.Finish());
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(stream.text(), "first");
}

TEST(TextStreamTest, ReportsCallbackException) {
  TextStream stream([](const std::string &) -> bool { throw std::runtime_error("cancel failed"); });
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(stream.Append("first"));
  const std::string error_log = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(error_log, "[2] Text stream callback failed: cancel failed\n");
  EXPECT_NE(stream.error_message().find("cancel failed"), std::string::npos);
}

} // namespace omni_runtime::inference
