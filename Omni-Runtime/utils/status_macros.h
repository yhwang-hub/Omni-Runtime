#pragma once

#include "Omni-Runtime/utils/status.h"

namespace omni_runtime::utils {

template <typename StatusType> inline bool IsStatusOk(const StatusType &status) {
  return status.ok();
}

} // namespace omni_runtime::utils

#define OMNI_RETURN_IF_STATUS_FAILED(status_expr, failure_value)                                   \
  do {                                                                                             \
    const auto omni_status = (status_expr);                                                        \
    if (!::omni_runtime::utils::IsStatusOk(omni_status)) {                                         \
      return failure_value;                                                                        \
    }                                                                                              \
  } while (false)

#define OMNI_RETURN_STATUS_IF_NOT_OK(status_expr)                                                       \
  OMNI_RETURN_IF_STATUS_FAILED(status_expr, omni_status)

#define OMNI_RETURN_FALSE_IF_NOT_OK(status_expr) OMNI_RETURN_IF_STATUS_FAILED(status_expr, false)
