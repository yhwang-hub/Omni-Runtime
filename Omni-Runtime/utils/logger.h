#pragma once

#include <cstdint>
#include <iostream>
#include <sstream>
#include <utility>

namespace omni_runtime::utils {

enum class LogLevel : uint8_t {
  kInfo = 0,
  kWarning = 1,
  kError = 2,
  kFatal = 3,
};

inline constexpr LogLevel INFO = LogLevel::kInfo;
inline constexpr LogLevel WARNING = LogLevel::kWarning;
inline constexpr LogLevel WARN = LogLevel::kWarning;
inline constexpr LogLevel ERROR = LogLevel::kError;
inline constexpr LogLevel FATAL = LogLevel::kFatal;

class LogLine final {
public:
  explicit LogLine(const LogLevel level) : stream_(level == ERROR || level == FATAL ? std::cerr : std::cout) {
    stream_ << "[" << static_cast<int>(level) << "] ";
  }

  ~LogLine() {
    stream_ << text_.str() << std::endl;
  }

  template <typename Value> LogLine &operator<<(Value &&value) {
    text_ << std::forward<Value>(value);
    return *this;
  }

private:
  std::ostringstream text_;
  std::ostream &stream_;
};

} // namespace omni_runtime::utils

using omni_runtime::utils::ERROR;
using omni_runtime::utils::FATAL;
using omni_runtime::utils::INFO;
using omni_runtime::utils::WARN;
using omni_runtime::utils::WARNING;

#define OMNI_RETURN_VAL_IF(condition, value)                                                            \
  do {                                                                                             \
    if (condition) {                                                                               \
      return value;                                                                                \
    }                                                                                              \
  } while (false)

#define OMNI_RETURN_VOID_IF(condition)                                                                  \
  do {                                                                                             \
    if (condition) {                                                                               \
      return;                                                                                      \
    }                                                                                              \
  } while (false)

#define OMNI_RETURN_IF_LOG(condition, level, message)                                                   \
  do {                                                                                             \
    if (condition) {                                                                               \
      ::omni_runtime::utils::LogLine(level) << message;                                            \
      return;                                                                                      \
    }                                                                                              \
  } while (false)

#define OMNI_RETURN_VAL_IF_LOG(condition, value, level, message)                                        \
  do {                                                                                             \
    if (condition) {                                                                               \
      ::omni_runtime::utils::LogLine(level) << message;                                            \
      return value;                                                                                \
    }                                                                                              \
  } while (false)

#define LOG_ERROR(message) ::omni_runtime::utils::LogLine(::omni_runtime::utils::ERROR) << message

#define LOG_INFO(message) ::omni_runtime::utils::LogLine(::omni_runtime::utils::INFO) << message
