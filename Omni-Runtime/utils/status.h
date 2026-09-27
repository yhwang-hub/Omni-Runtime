#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>

namespace omni_runtime::utils {

enum class StatusCode : uint8_t {
  kOk,
  kCancelled,
  kInvalidArgument,
  kNotFound,
  kResourceExhausted,
  kDeviceError,
  kKernelError,
  kIOError,
  kInternal,
  kUnimplemented,
};

struct SourceLocation final {
  const char *file = nullptr;
  int line = 0;
  const char *function = nullptr;

  SourceLocation() = default;
  explicit SourceLocation(const char *file, const int line, const char *function)
      : file(file), line(line), function(function) {
  }
};

class Status final {
public:
  Status() = default;
  explicit Status(const StatusCode code, const std::string_view &message, const SourceLocation &location)
      : code_(code), message_(message), location_(location) {
  }

  bool ok() const {
    return code_ == StatusCode::kOk;
  }

  StatusCode code() const {
    return code_;
  }

  const std::string &message() const {
    return message_;
  }

  SourceLocation location() const {
    return location_;
  }

  static Status OK() {
    return Status();
  }

  static Status Cancelled(const std::string_view &message, const SourceLocation &location) {
    return Create<StatusCode::kCancelled>(message, location);
  }

  static Status InvalidArgument(const std::string_view &message, const SourceLocation &location) {
    return Create<StatusCode::kInvalidArgument>(message, location);
  }

  static Status NotFound(const std::string_view &message, const SourceLocation &location) {
    return Create<StatusCode::kNotFound>(message, location);
  }

  static Status ResourceExhausted(const std::string_view &message, const SourceLocation &location) {
    return Create<StatusCode::kResourceExhausted>(message, location);
  }

  static Status DeviceError(const std::string_view &message, const SourceLocation &location) {
    return Create<StatusCode::kDeviceError>(message, location);
  }

  static Status KernelError(const std::string_view &message, const SourceLocation &location) {
    return Create<StatusCode::kKernelError>(message, location);
  }

  static Status IOError(const std::string_view &message, const SourceLocation &location) {
    return Create<StatusCode::kIOError>(message, location);
  }

  static Status Internal(const std::string_view &message, const SourceLocation &location) {
    return Create<StatusCode::kInternal>(message, location);
  }

  static Status Unimplemented(const std::string_view &message, const SourceLocation &location) {
    return Create<StatusCode::kUnimplemented>(message, location);
  }

private:
  template <StatusCode code> static Status Create(const std::string_view &message, const SourceLocation &location) {
    return Status(code, message, location);
  }

  StatusCode code_ = StatusCode::kOk;
  std::string message_;
  SourceLocation location_;
};

inline std::ostream &operator<<(std::ostream &stream, const Status &status) {
  return stream << "code=" << static_cast<int>(status.code()) << ", message=" << status.message();
}

} // namespace omni_runtime::utils

#define SOURCE_LOCATION() ::omni_runtime::utils::SourceLocation(__FILE__, __LINE__, __func__)
