#pragma once

#include <type_traits>

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "Omni-Runtime/backend/opencl_jit/executor.h"
#include "kernels/operator_runtime_common.h"

namespace omni_runtime::kernels::runtime {

template <typename Operation> struct OpenCLOperator;

inline bool ConfigureOpenCL(const std::string &source, const std::string &entry,
                            const uint64_t groups, const size_t local_size,
                            backend::OpenCLInvocation *const invocation) {
  OMNI_RETURN_VAL_IF_LOG(invocation == nullptr || groups == 0U || local_size == 0U ||
                             groups > std::numeric_limits<size_t>::max() / local_size,
                         false, ERROR, "Invalid OpenCL operator dimensions");
  invocation->source = source;
  invocation->entry = entry;
  invocation->global_size = {static_cast<size_t>(groups) * local_size, 1U, 1U};
  invocation->local_size = {local_size, 1U, 1U};
  return true;
}

template <typename T>
void BindOpenCLArgument(const T &value, backend::OpenCLInvocation *const invocation) {
  if constexpr (std::is_same_v<T, backend::DeviceView>) {
    invocation->Buffer(value);
  } else if constexpr (std::is_same_v<T, bool>) {
    invocation->Scalar(static_cast<int32_t>(value));
  } else {
    invocation->Scalar(value);
  }
}

template <typename... Args>
void BindOpenCLArguments(backend::OpenCLInvocation *const invocation, const Args &...args) {
  (BindOpenCLArgument(args, invocation), ...);
}

} // namespace omni_runtime::kernels::runtime
