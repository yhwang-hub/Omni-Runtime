#pragma once

#include <type_traits>

#include "Omni-Runtime/backend/backend_compute_define.h"
#include "kernels/cuda_operator_runtime.h"
#include "kernels/opencl_operator_runtime.h"

namespace omni_runtime::kernels::runtime {

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
    invocation.compute_unit_count = executor->compute_unit_count();
    OMNI_RETURN_VAL_IF(!OpenCLOperator<Operation>::Make(operation, &invocation), false);
    if constexpr (requires { OpenCLOperator<Operation>::Run(invocation, executor); }) {
      return OpenCLOperator<Operation>::Run(invocation, executor);
    }
    return executor->Launch(invocation);
  }
};

} // namespace omni_runtime::kernels::runtime
