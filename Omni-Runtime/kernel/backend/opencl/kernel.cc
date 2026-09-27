#include "Omni-Runtime/kernel/backend/opencl/kernel.h"

#include <fstream>
#include <iterator>
#include <string>

#include "Omni-Runtime/kernel/utils/exception.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace opencl {
namespace {

std::string ReadSource(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  JIT_HOST_ASSERT(stream.is_open(), "failed to open OpenCL source: {}", path.string());
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

} // namespace

bool Kernel::Load(cl_context context, cl_device_id device, const std::filesystem::path &source_path,
                  const std::string &entry_name, const CompilerOptions &options,
                  std::shared_ptr<Kernel> *const kernel) {
  JIT_HOST_ASSERT(kernel != nullptr, "kernel output pointer is null");
  const std::string source = ReadSource(source_path);
  const char *source_data = source.c_str();
  const std::size_t source_length = source.size();
  cl_int status = CL_SUCCESS;
  cl_program program =
      clCreateProgramWithSource(context, 1U, &source_data, &source_length, &status);
  JIT_HOST_ASSERT(status == CL_SUCCESS, "failed to create OpenCL program: {}", status);

  std::string build_options;
  for (const std::string &option_flag : options.get_flags()) {
    if (!build_options.empty()) {
      build_options += ' ';
    }
    build_options += option_flag;
  }
  status =
      clBuildProgram(program, 1U, &device, build_options.empty() ? nullptr : build_options.c_str(),
                     nullptr, nullptr);
  if (status != CL_SUCCESS) {
    std::size_t log_size = 0U;
    static_cast<void>(
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0U, nullptr, &log_size));
    std::string log;
    if (log_size > 0U) {
      std::vector<char> log_buffer(log_size);
      static_cast<void>(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                                              log_buffer.data(), nullptr));
      log.assign(log_buffer.data(), log_size);
    }
    clReleaseProgram(program);
    JIT_PANIC("failed to build OpenCL program: {}: {}", status, log);
  }

  cl_kernel native_kernel = clCreateKernel(program, entry_name.c_str(), &status);
  if (status != CL_SUCCESS) {
    clReleaseProgram(program);
    JIT_PANIC("failed to create OpenCL kernel {}: {}", entry_name, status);
  }

  *kernel = std::shared_ptr<Kernel>(new Kernel(context, program, native_kernel));
  return true;
}

Kernel::Kernel(cl_context context, cl_program program, cl_kernel kernel)
    : context_(context), program_(program), kernel_(kernel) {
  if (context_ != nullptr) {
    static_cast<void>(clRetainContext(context_));
  }
}

Kernel::~Kernel() {
  if (kernel_ != nullptr) {
    static_cast<void>(clReleaseKernel(kernel_));
  }
  if (program_ != nullptr) {
    static_cast<void>(clReleaseProgram(program_));
  }
  if (context_ != nullptr) {
    static_cast<void>(clReleaseContext(context_));
  }
}

bool Kernel::Launch(cl_command_queue queue, const std::vector<std::size_t> &global_size,
                    const std::vector<std::size_t> &local_size,
                    const std::vector<const void *> &arguments) const {
  OMNI_RETURN_VAL_IF(kernel_ == nullptr || queue == nullptr, false);
  OMNI_RETURN_VAL_IF(global_size.empty() || global_size.size() > 3U, false);
  OMNI_RETURN_VAL_IF(local_size.size() != global_size.size(), false);
  for (const void *argument : arguments) {
    OMNI_RETURN_VAL_IF(argument == nullptr, false);
  }
  for (std::size_t index = 0; index < arguments.size(); ++index) {
    if (clSetKernelArg(kernel_, static_cast<cl_uint>(index), sizeof(void *), &arguments[index]) !=
        CL_SUCCESS) {
      return false;
    }
  }
  std::size_t global_work_size[3] = {1U, 1U, 1U};
  std::size_t local_work_size[3] = {1U, 1U, 1U};
  for (std::size_t index = 0; index < global_size.size() && index < 3U; ++index) {
    global_work_size[index] = global_size[index];
    local_work_size[index] = local_size[index];
  }
  return clEnqueueNDRangeKernel(queue, kernel_, static_cast<cl_uint>(global_size.size()), nullptr,
                                global_work_size, local_work_size, 0U, nullptr,
                                nullptr) == CL_SUCCESS;
}

cl_kernel Kernel::native_kernel() const {
  return kernel_;
}

} // namespace opencl
} // namespace backend
} // namespace jit
} // namespace omni_runtime
