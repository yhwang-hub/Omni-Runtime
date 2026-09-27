#pragma once

#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "Omni-Runtime/backend/cuda_jit/jit_runtime.h"
#include "Omni-Runtime/backend/cuda_jit/launch_define.h"
#include "Omni-Runtime/utils/logger.h"
#include "kernels/kernel_define.h"

namespace omni_runtime {
namespace kernels {
namespace runtime {

class CudaKernelRuntime final {
public:
  static bool HasTensorMma() {
    return kernel::jit::JitRuntime::HasTensorMma();
  }

  static bool HasSm120() {
    return kernel::jit::JitRuntime::HasSm120();
  }

  static int32_t MultiprocessorCount() {
    return kernel::jit::JitRuntime::MultiprocessorCount();
  }

  template <typename... KernelArgs>
  static bool CompileAndLaunch(const std::string &tag, const std::string &source,
                               const KernelLaunchConfig &options,
                               const KernelArgs &...kernel_args) {
    OMNI_RETURN_VAL_IF(tag.empty() || source.empty(), false);

    std::shared_ptr<kernel::jit::CudaKernel> kernel;
    OMNI_RETURN_VAL_IF(!kernel::jit::JitRuntime::Compile(tag, source, &kernel), false);
    return kernel::jit::JitRuntime::Launch(kernel, options.grid_dim, options.block_dim,
                                           options.shared_memory_bytes, kernel_args...);
  }

  static std::string MakeKernelSource(const std::string &include_path,
                                      const std::string &kernel_name) {
    return std::format(R"KERNEL_SOURCE(
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <{}>
static void __instantiate_kernel() {{
    (void)&{};
}}
)KERNEL_SOURCE",
                       include_path, kernel_name);
  }

  template <typename T, int... Values>
  static std::string MakeKernelSource(const std::string &include_path,
                                      const std::string &kernel_name) {
    std::string parameters;
    ((parameters += ", " + std::to_string(Values)), ...);
    return std::format(R"KERNEL_SOURCE(
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <{}>
static void __instantiate_kernel() {{
    (void)&{}<{}{}>;
}}
)KERNEL_SOURCE",
                       include_path, kernel_name, TypeName<T>(), parameters);
  }

private:
  template <typename T> static constexpr std::string_view TypeName() {
    if constexpr (std::is_same_v<T, float>) {
      return "float";
    } else if constexpr (std::is_same_v<T, double>) {
      return "double";
    } else if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, Float16>) {
      return "__half";
    } else if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, BFloat16>) {
      return "__nv_bfloat16";
    } else {
      static_assert(sizeof(T) == 0, "unsupported kernel element type");
    }
  }
};

} // namespace runtime
} // namespace kernels
} // namespace omni_runtime
