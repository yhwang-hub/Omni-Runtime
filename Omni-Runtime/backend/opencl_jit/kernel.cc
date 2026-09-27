#include "Omni-Runtime/backend/opencl_jit/kernel.h"

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

Program::~Program() {
  if (program_ != nullptr && clReleaseProgram(program_) != CL_SUCCESS) {
    LOG_ERROR("Failed to release an OpenCL program");
  }
}

cl_program Program::native_program() const {
  return program_;
}

bool Kernel::BuildProgram(const cl_context context, const cl_device_id device,
                          const std::string &source, const CompilerOptions &options,
                          std::shared_ptr<Program> *const output) {
  OMNI_RETURN_VAL_IF_LOG(context == nullptr || device == nullptr || output == nullptr ||
                             source.empty(),
                         false, ERROR, "Invalid OpenCL program inputs");
  const char *source_data = source.c_str();
  const std::size_t source_length = source.size();
  cl_int status = CL_SUCCESS;
  cl_program program =
      clCreateProgramWithSource(context, 1U, &source_data, &source_length, &status);
  OMNI_RETURN_VAL_IF_LOG(status != CL_SUCCESS || program == nullptr, false, ERROR,
                         "Failed to create OpenCL program: " << status);
  auto compiled = std::make_shared<Program>();
  compiled->program_ = program;

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
      if (!log.empty() && log.back() == '\0') {
        log.pop_back();
      }
    }
    LOG_ERROR("Failed to build OpenCL program: " << status << ": " << log);
    return false;
  }
  *output = std::move(compiled);
  return true;
}

bool Kernel::Load(cl_context context, cl_device_id device, const std::filesystem::path &source_path,
                  const std::string &entry_name, const CompilerOptions &options,
                  std::shared_ptr<Kernel> *const kernel) {
  OMNI_RETURN_VAL_IF_LOG(kernel == nullptr, false, ERROR, "OpenCL kernel output is null");
  std::shared_ptr<Program> program;
  OMNI_RETURN_VAL_IF(!BuildProgram(context, device, ReadSource(source_path), options, &program),
                     false);
  cl_int status = CL_SUCCESS;
  cl_kernel native_kernel = clCreateKernel(program->native_program(), entry_name.c_str(), &status);
  OMNI_RETURN_VAL_IF_LOG(status != CL_SUCCESS || native_kernel == nullptr, false, ERROR,
                         "Failed to create OpenCL kernel " << entry_name << ": " << status);
  *kernel = std::shared_ptr<Kernel>(new Kernel(context, program->native_program(), native_kernel));
  return true;
}

Kernel::Kernel(cl_context context, cl_program program, cl_kernel kernel)
    : context_(context), program_(program), kernel_(kernel) {
  if (context_ != nullptr) {
    static_cast<void>(clRetainContext(context_));
  }
  if (program_ != nullptr) {
    static_cast<void>(clRetainProgram(program_));
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
  std::vector<::omni_runtime::backend::OpenCLArgument> typed_arguments;
  for (const void *const argument : arguments) {
    OMNI_RETURN_VAL_IF_LOG(argument == nullptr, false, ERROR, "Null legacy OpenCL argument");
    typed_arguments.push_back(
        {.kind = ::omni_runtime::backend::OpenCLArgument::Kind::kBuffer,
         .buffer = {.data = const_cast<void *>(argument),
                    .size_bytes = 1U,
                    .backend_kind = ::omni_runtime::backend::BackendKind::kOpenCL}});
  }
  return Launch(queue, global_size, local_size, typed_arguments);
}

bool Kernel::BindArguments(const cl_kernel kernel,
                           const std::vector<::omni_runtime::backend::OpenCLArgument> &arguments) {
  OMNI_RETURN_VAL_IF_LOG(kernel == nullptr, false, ERROR, "OpenCL kernel is null");
  using Argument = ::omni_runtime::backend::OpenCLArgument;
  for (size_t index = 0U; index < arguments.size(); ++index) {
    const Argument &argument = arguments[index];
    cl_int status = CL_SUCCESS;
    if (argument.kind == Argument::Kind::kBuffer) {
      OMNI_RETURN_VAL_IF_LOG(
          argument.buffer.data != nullptr &&
              (!argument.buffer.IsValid() ||
               argument.buffer.backend_kind != ::omni_runtime::backend::BackendKind::kOpenCL),
          false, ERROR, "Invalid or cross-backend OpenCL buffer argument");
      const auto memory = static_cast<cl_mem>(argument.buffer.data);
      status = clSetKernelArg(kernel, index, sizeof(memory), &memory);
    } else if (argument.kind == Argument::Kind::kScalar) {
      OMNI_RETURN_VAL_IF_LOG(argument.size_bytes == 0U ||
                                 argument.size_bytes > argument.value.size(),
                             false, ERROR, "Invalid OpenCL scalar argument size");
      status = clSetKernelArg(kernel, index, argument.size_bytes, argument.value.data());
    } else {
      OMNI_RETURN_VAL_IF_LOG(argument.size_bytes == 0U, false, ERROR,
                             "OpenCL local argument is empty");
      status = clSetKernelArg(kernel, index, argument.size_bytes, nullptr);
    }
    OMNI_RETURN_VAL_IF_LOG(status != CL_SUCCESS, false, ERROR,
                           "OpenCL argument " << index << " binding failed: " << status);
  }
  return true;
}

bool Kernel::Launch(const cl_command_queue queue, const std::vector<std::size_t> &global_size,
                    const std::vector<std::size_t> &local_size,
                    const std::vector<::omni_runtime::backend::OpenCLArgument> &arguments) const {
  std::lock_guard<std::mutex> lock(mutex_);
  OMNI_RETURN_VAL_IF(kernel_ == nullptr || queue == nullptr, false);
  OMNI_RETURN_VAL_IF(global_size.empty() || global_size.size() > 3U, false);
  OMNI_RETURN_VAL_IF(local_size.size() != global_size.size(), false);
  OMNI_RETURN_VAL_IF(!BindArguments(kernel_, arguments), false);
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
