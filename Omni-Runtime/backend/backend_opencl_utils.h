#pragma once

#include <limits>
#include <tuple>
#include <type_traits>

#include "Omni-Runtime/backend/opencl_jit/executor.h"

namespace omni_runtime::backend::opencl_internal {

inline size_t RoundWorkSize(const uint64_t elements, const size_t local_size) {
  return static_cast<size_t>((elements + local_size - 1U) / local_size) * local_size;
}

inline bool ConfigureOpenCL(const std::string &source, const std::string &entry,
                            const uint64_t groups, const size_t local_size,
                            OpenCLInvocation *const invocation) {
  OMNI_RETURN_VAL_IF_LOG(invocation == nullptr || groups == 0U || local_size == 0U ||
                             groups > std::numeric_limits<size_t>::max() / local_size,
                         false, ERROR, "Invalid OpenCL operator dimensions");
  invocation->source = source;
  invocation->entry = entry;
  invocation->global_size = {static_cast<size_t>(groups) * local_size, 1U, 1U};
  invocation->local_size = {local_size, 1U, 1U};
  return true;
}

template <typename T> void BindOpenCLArgument(const T &value, OpenCLInvocation *const invocation) {
  if constexpr (std::is_same_v<T, DeviceView>) {
    invocation->Buffer(value);
  } else if constexpr (std::is_same_v<T, bool>) {
    invocation->Scalar(static_cast<int32_t>(value));
  } else {
    invocation->Scalar(value);
  }
}

template <typename... Args>
bool BindOpenCLArguments(const std::tuple<Args...> &arguments, OpenCLInvocation *const invocation) {
  OMNI_RETURN_VAL_IF(invocation == nullptr, false);
  std::apply([invocation](const auto &...args) { (BindOpenCLArgument(args, invocation), ...); },
             arguments);
  return true;
}

} // namespace omni_runtime::backend::opencl_internal
