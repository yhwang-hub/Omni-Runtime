#include "Omni-Runtime/kernel/backend/opencl/backend.h"

#include <algorithm>
#include <fstream>
#include <vector>

#include "Omni-Runtime/kernel/utils/exception.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace opencl {

struct OpenCL::Runtime final {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
};

OpenCL::OpenCL() : runtime_(std::make_unique<Runtime>()) {
  std::vector<cl_platform_id> platforms;
  std::uint32_t platform_count = 0U;
  if (clGetPlatformIDs(0U, nullptr, &platform_count) != CL_SUCCESS || platform_count == 0U) {
    return;
  }
  platforms.resize(platform_count);
  if (clGetPlatformIDs(platform_count, platforms.data(), nullptr) != CL_SUCCESS) {
    return;
  }
  for (const cl_platform_id platform : platforms) {
    std::uint32_t device_count = 0U;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0U, nullptr, &device_count) != CL_SUCCESS ||
        device_count == 0U) {
      continue;
    }
    std::vector<cl_device_id> devices(device_count);
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr) !=
        CL_SUCCESS) {
      continue;
    }
    runtime_->platform = platform;
    runtime_->device = devices.front();
    break;
  }
  if (runtime_->device == nullptr) {
    for (const cl_platform_id platform : platforms) {
      std::uint32_t device_count = 0U;
      if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0U, nullptr, &device_count) != CL_SUCCESS ||
          device_count == 0U) {
        continue;
      }
      std::vector<cl_device_id> devices(device_count);
      if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, device_count, devices.data(), nullptr) !=
          CL_SUCCESS) {
        continue;
      }
      runtime_->platform = platform;
      runtime_->device = devices.front();
      break;
    }
  }
  if (runtime_->device == nullptr) {
    return;
  }
  runtime_->context = clCreateContext(nullptr, 1U, &runtime_->device, nullptr, nullptr, nullptr);
}

OpenCL::~OpenCL() {
  if (runtime_ == nullptr) {
    return;
  }
  if (runtime_->context != nullptr) {
    static_cast<void>(clReleaseContext(runtime_->context));
  }
}

bool OpenCL::IsAvailable() const {
  return runtime_ != nullptr && runtime_->context != nullptr;
}

OpenCL::Device OpenCL::device() const {
  return runtime_->device;
}

cl_context OpenCL::context() const {
  return runtime_->context;
}

OpenCL::CompilerOptions OpenCL::default_compiler_options() {
  return CompilerOptions::default_options(utils::Env("OMNI"));
}

bool OpenCL::Compile(const std::string &source, const std::filesystem::path &source_path,
                     const std::string &entry_name, const CompilerOptions &options,
                     std::shared_ptr<Kernel> *const kernel) {
  JIT_HOST_ASSERT(kernel != nullptr, "kernel output pointer is null");
  JIT_HOST_ASSERT(IsAvailable(), "OpenCL runtime is unavailable");
  std::error_code error;
  std::filesystem::create_directories(source_path.parent_path(), error);
  JIT_HOST_ASSERT(!error, "failed to create OpenCL JIT directory: {}", source_path.string());
  std::ofstream stream(source_path, std::ios::binary | std::ios::trunc);
  JIT_HOST_ASSERT(stream.is_open(), "failed to write OpenCL source: {}", source_path.string());
  stream.write(source.data(), static_cast<std::streamsize>(source.size()));
  stream.close();
  JIT_HOST_ASSERT(stream.good(), "failed to flush OpenCL source: {}", source_path.string());
  return Kernel::Load(runtime_->context, runtime_->device, source_path, entry_name, options,
                      kernel);
}

} // namespace opencl
} // namespace backend
} // namespace jit
} // namespace omni_runtime
