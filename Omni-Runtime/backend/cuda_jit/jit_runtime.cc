#include "Omni-Runtime/backend/cuda_jit/jit_runtime.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <mutex>
#include <unordered_map>

#include "Omni-Runtime/backend/cuda_jit/backend.h"
#include "Omni-Runtime/kernel/runtime/runtime.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace kernel {
namespace jit {
namespace {

std::mutex compile_mutex;
std::unordered_map<std::string, std::shared_ptr<CudaKernel>> compiled_kernels;

} // namespace

using RuntimeType =
    ::omni_runtime::jit::backend::runtime::Runtime<::omni_runtime::jit::backend::cuda::CUDA>;
using LazyRuntimeType = ::omni_runtime::jit::utils::lazy::LazyInit<RuntimeType>;

thread_local const void *JitRuntime::stream_ = nullptr;
thread_local bool JitRuntime::is_preparing_ = false;
thread_local std::vector<std::shared_ptr<CudaKernel>> *JitRuntime::kernels_ = nullptr;

JitRuntime::LaunchScope::LaunchScope(const void *const stream, const bool is_preparing,
                                     std::vector<std::shared_ptr<CudaKernel>> *const kernels)
    : previous_stream_(stream_), was_preparing_(is_preparing_), previous_kernels_(kernels_) {
  stream_ = stream;
  is_preparing_ = is_preparing;
  kernels_ = kernels;
}

JitRuntime::LaunchScope::~LaunchScope() {
  stream_ = previous_stream_;
  is_preparing_ = was_preparing_;
  kernels_ = previous_kernels_;
}

bool JitRuntime::Init(const std::filesystem::path &library_root) {
  OMNI_RETURN_VAL_IF(library_root.empty(), false);
  std::error_code error;
  const std::filesystem::path absolute_root = std::filesystem::absolute(library_root, error);
  OMNI_RETURN_VAL_IF(error || !std::filesystem::is_directory(absolute_root, error), false);
  const std::lock_guard<std::mutex> lock(compile_mutex);
  compiled_kernels.clear();
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
  const std::lock_guard<std::mutex> lock(compile_mutex);
  const auto iterator = compiled_kernels.find(source);
  if (iterator != compiled_kernels.end()) {
    *kernel = iterator->second;
    return true;
  }
  try {
    *kernel = Runtime().compile(tag, source, {});
  } catch (const std::exception &error) {
    std::cerr << "JIT compile error: " << error.what() << std::endl;
    return false;
  } catch (...) {
    std::cerr << "JIT compile unknown error" << std::endl;
    return false;
  }
  OMNI_RETURN_VAL_IF(*kernel == nullptr, false);
  compiled_kernels.emplace(source, *kernel);
  return true;
}

bool JitRuntime::HasTensorMma() {
  const std::lock_guard<std::mutex> lock(compile_mutex);
  return Runtime().device.get_arch_major() >= 8;
}

bool JitRuntime::HasSm120() {
  const std::lock_guard<std::mutex> lock(compile_mutex);
  return Runtime().device.get_arch_major() == 12 && Runtime().device.get_arch_minor() <= 1;
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
