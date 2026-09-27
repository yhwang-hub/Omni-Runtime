#pragma once

#include <type_traits>

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "Omni-Runtime/backend/opencl_jit/executor.h"
#include "Omni-Runtime/kernels/kernel_define.h"

namespace omni_runtime::kernels::runtime {

template <typename T> T *CudaAddress(const backend::DeviceView &view) {
  if (!view.IsValid()) {
    return nullptr;
  }
  OMNI_RETURN_VAL_IF_LOG(view.backend_kind != backend::BackendKind::kCuda, nullptr, ERROR,
                         "A CUDA operator received memory from a different backend");
  return reinterpret_cast<T *>(static_cast<uint8_t *>(view.data) + view.offset_bytes);
}

template <typename T, typename Operation> struct CudaOperator;
template <typename Operation> struct OpenCLOperator;

template <typename T> constexpr backend::ScalarType OperatorScalarType() {
  if constexpr (std::is_same_v<T, float>) {
    return backend::ScalarType::kFloat32;
  } else if constexpr (std::is_same_v<T, Float16>) {
    return backend::ScalarType::kFloat16;
  } else {
    static_assert(std::is_same_v<T, BFloat16>);
    return backend::ScalarType::kBFloat16;
  }
}

template <typename T, typename Operation> struct OperatorRuntime final {
  static bool Launch(const backend::BackendKind backend_kind, const Operation &operation,
                     backend::OpenCLExecutor *const executor) {
    if (backend_kind == backend::BackendKind::kCuda) {
      if constexpr (std::is_same_v<Operation, backend::Linear>) {
        LOG_ERROR("CUDA linear execution requires its backend-owned BLAS service");
        return false;
      } else {
        return CudaOperator<T, Operation>::Launch(operation);
      }
    }
    OMNI_RETURN_VAL_IF_LOG(backend_kind != backend::BackendKind::kOpenCL || executor == nullptr,
                           false, ERROR, "Invalid operator execution backend");
    backend::OpenCLInvocation invocation;
    invocation.type = OperatorScalarType<T>();
    invocation.has_nvidia_mma = executor->HasNvidiaMma();
    OMNI_RETURN_VAL_IF(!OpenCLOperator<Operation>::Make(operation, &invocation), false);
    if constexpr (requires { OpenCLOperator<Operation>::Run(invocation, executor); }) {
      return OpenCLOperator<Operation>::Run(invocation, executor);
    }
    return executor->Launch(invocation);
  }
};

inline size_t RoundWorkSize(const uint64_t elements, const size_t local_size) {
  return static_cast<size_t>((elements + local_size - 1U) / local_size) * local_size;
}

inline bool HasElements(const backend::DeviceView &view, const uint64_t count,
                        const size_t element_bytes) {
  return view.IsValid() && element_bytes != 0U && view.offset_bytes % element_bytes == 0U &&
         count <= view.size_bytes / element_bytes;
}

inline size_t ElementBytes(const backend::ScalarType type) {
  return type == backend::ScalarType::kFloat32 ? sizeof(float) : sizeof(uint16_t);
}

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
