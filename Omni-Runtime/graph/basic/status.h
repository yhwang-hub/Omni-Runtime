#pragma once

#include <string>

#include "Omni-Runtime/utils/status.h"

namespace omni_runtime {
namespace graph {

using Status = omni_runtime::utils::Status;
using StatusCode = omni_runtime::utils::StatusCode;

inline Status InvalidArgumentStatus(const std::string &message) {
  return Status::InvalidArgument(message, SOURCE_LOCATION());
}

inline Status InternalStatus(const std::string &message) {
  return Status::Internal(message, SOURCE_LOCATION());
}

inline Status NotFoundStatus(const std::string &message) {
  return Status::NotFound(message, SOURCE_LOCATION());
}

inline Status UnimplementedStatus(const std::string &message) {
  return Status::Unimplemented(message, SOURCE_LOCATION());
}

} // namespace graph
} // namespace omni_runtime
