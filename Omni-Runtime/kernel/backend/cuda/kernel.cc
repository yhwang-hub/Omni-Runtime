#include "Omni-Runtime/kernel/backend/cuda/kernel.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace cuda {

std::shared_ptr<Kernel> Kernel::load(const std::filesystem::path &dir,
                                     const ::omni_runtime::jit::utils::Env &env) {
  // Release GIL to let other Python threads run

  // Check existence
  const auto cubin_path = dir / "kernel.cubin";
  if (not std::filesystem::is_regular_file(cubin_path)) {
    JIT_PANIC("missing CUDA CUBIN: {}", cubin_path.string());
  }

  // Record start time
  const bool debug = env.get<bool>("JIT_DEBUG", false);
  const bool print_load_time = debug or env.get<bool>("JIT_PRINT_LOAD_TIME", false);
  if (debug) {
    std::fputs(std::format("Loading CUBIN: {}\n", cubin_path.string()).c_str(), stdout);
  }
  const auto start_time = std::chrono::steady_clock::now();

  // Load kernel
  CUlibrary library_handle{};
  CUfunction kernel_handle{};
  JIT_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryLoadFromFile(
      &library_handle, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
  try {
    unsigned int num_kernels = 0;
    JIT_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryGetKernelCount(&num_kernels, library_handle));
    if (num_kernels != 1) {
      JIT_PANIC("expected exactly one kernel in {}", cubin_path.string());
    }

    CUkernel library_kernel_handle{};
    JIT_CUDA_DRIVER_CHECK(
        driver::lazy_cuLibraryEnumerateKernels(&library_kernel_handle, 1, library_handle));
    JIT_CUDA_DRIVER_CHECK(driver::lazy_cuKernelGetFunction(&kernel_handle, library_kernel_handle));
  } catch (...) {
    driver::lazy_cuLibraryUnload(library_handle);
    throw;
  }

  // Print and return
  if (print_load_time) {
    const std::chrono::duration<double, std::milli> elapsed =
        std::chrono::steady_clock::now() - start_time;
    std::fputs(std::format("Load time ({}): {:.2f} ms\n", dir.string(), elapsed.count()).c_str(),
               stdout);
  }
  return std::make_shared<Kernel>(library_handle, kernel_handle);
}

void Kernel::unload() noexcept {
  if (library_handle == nullptr) {
    return;
  }

  try {
    JIT_CUDA_DRIVER_CHECK(driver::lazy_cuLibraryUnload(library_handle));
  } catch (...) {
  }
  library_handle = nullptr;
  kernel_handle = nullptr;
}

} // namespace cuda
} // namespace backend
} // namespace jit
} // namespace omni_runtime
