#include "Omni-Runtime/kernel/runtime/jit.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "Omni-Runtime/kernel/backend/cuda/backend.h"
#include "Omni-Runtime/kernel/runtime/runtime.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace kernel {
namespace jit {

using RuntimeType =
    ::omni_runtime::jit::backend::runtime::Runtime<::omni_runtime::jit::backend::cuda::CUDA>;
using LazyRuntimeType = ::omni_runtime::jit::utils::lazy::LazyInit<RuntimeType>;

bool JitRuntime::Init(const std::filesystem::path &library_root) {
  OMNI_RETURN_VAL_IF(library_root.empty(), false);
  std::error_code error;
  const std::filesystem::path absolute_root = std::filesystem::absolute(library_root, error);
  OMNI_RETURN_VAL_IF(error || !std::filesystem::is_directory(absolute_root, error), false);
  const ::omni_runtime::jit::backend::runtime::Config config(
      absolute_root, "OMNI", "omni-kernels-v1", {absolute_root}, {"kernels/"});
  LazyRuntime() = LazyRuntimeType([config] {
    auto runtime = std::make_shared<RuntimeType>(config);
    runtime->default_compiler_options.nvcc_flags->emplace_back("--expt-relaxed-constexpr");
    runtime->default_compiler_options.nvcc_flags->emplace_back("--expt-extended-lambda");
    runtime->default_compiler_options.nvcc_flags->emplace_back(
        "--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
    return runtime;
  });
  return true;
}

bool JitRuntime::Compile(const std::string &tag, const std::string &source,
                         std::shared_ptr<CudaKernel> *const kernel) {
  OMNI_RETURN_VAL_IF(kernel == nullptr || tag.empty() || source.empty(), false);
  try {
    *kernel = Runtime().compile(tag, source, {});
  } catch (const std::exception &error) {
    std::cerr << "JIT compile error: " << error.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "JIT compile unknown error" << std::endl;
    return false;
  }
  return *kernel != nullptr;
}

RuntimeType &JitRuntime::Runtime() {
  return *LazyRuntime().operator->();
}

LazyRuntimeType &JitRuntime::LazyRuntime() {
  static LazyRuntimeType lazy_runtime(nullptr);
  return lazy_runtime;
}

} // namespace jit
} // namespace kernel
} // namespace omni_runtime
