#include "Omni-Runtime/backend/opencl_jit/executor.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <fstream>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <unordered_map>

#include "Omni-Runtime/backend/runtime_path.h"
#include "Omni-Runtime/backend/opencl_jit/kernel.h"

namespace omni_runtime::backend {
namespace opencl = ::omni_runtime::jit::backend::opencl;
namespace {

bool ReadSource(const std::filesystem::path &path, std::string *const source) {
  OMNI_RETURN_VAL_IF(source == nullptr, false);
  std::ifstream input(path, std::ios::binary);
  OMNI_RETURN_VAL_IF_LOG(!input.is_open(), false, ERROR,
                         "Cannot open OpenCL kernel source: " << path.string());
  source->assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  return input.eof() || !input.fail();
}

} // namespace

class OpenCLExecutor::PreparedKernel final {
public:
  ~PreparedKernel() {
    if (kernel_ != nullptr && clReleaseKernel(kernel_) != CL_SUCCESS) {
      LOG_ERROR("Failed to release a prepared OpenCL kernel");
    }
  }

private:
  friend class OpenCLExecutor;
  cl_kernel kernel_ = nullptr;
  std::array<size_t, 3U> global_size_ = {};
  std::array<size_t, 3U> local_size_ = {};
  uint32_t dimensions_ = 0U;
  std::string entry_;
};

struct OpenCLExecutor::Runtime final {
  cl_context context = nullptr;
  cl_device_id device = nullptr;
  cl_command_queue queue = nullptr;
  std::filesystem::path root;
  std::string prelude;
  bool has_nvidia_warp = false;
  bool has_nvidia_mma = false;
  bool has_correctly_rounded_division = false;
  size_t max_work_group_size = 0U;
  cl_ulong local_memory_bytes = 0U;
  bool enable_profiling = false;
  std::vector<std::pair<std::string, cl_event>> samples;
  std::unordered_map<std::string, std::shared_ptr<opencl::Program>> programs;
  PreparedSequence *preparation = nullptr;
  std::mutex mutex;

  ~Runtime() {
    if (!samples.empty()) {
      if (clFinish(queue) != CL_SUCCESS) {
        LOG_ERROR("Failed to complete OpenCL profiling queue");
      }
      std::unordered_map<std::string, std::pair<size_t, double>> totals;
      for (const auto &[entry, event] : samples) {
        cl_ulong start_ns = 0U;
        cl_ulong end_ns = 0U;
        if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_ns), &start_ns,
                                    nullptr) == CL_SUCCESS &&
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_ns), &end_ns,
                                    nullptr) == CL_SUCCESS &&
            end_ns >= start_ns) {
          auto &total = totals[entry];
          ++total.first;
          total.second += static_cast<double>(end_ns - start_ns) / 1000000.0;
        } else {
          LOG_ERROR("Cannot read OpenCL profiling event: " << entry);
        }
        static_cast<void>(clReleaseEvent(event));
      }
      for (const auto &[entry, total] : totals) {
        LOG_INFO("OpenCL profile " << entry << ": calls=" << total.first
                                   << ", total_ms=" << total.second);
      }
    }
    programs.clear();
    if (queue != nullptr) {
      static_cast<void>(clReleaseCommandQueue(queue));
    }
    if (context != nullptr) {
      static_cast<void>(clReleaseContext(context));
    }
  }
};

OpenCLExecutor::OpenCLExecutor() : runtime_(std::make_unique<Runtime>()) {
}

OpenCLExecutor::~OpenCLExecutor() = default;

bool OpenCLExecutor::Initialize(void *const context, void *const device, void *const queue) {
  OMNI_RETURN_VAL_IF_LOG(context == nullptr || device == nullptr || queue == nullptr, false, ERROR,
                         "Invalid OpenCL execution environment");
  OMNI_RETURN_VAL_IF(runtime_->context != nullptr, true);
  const auto native_context = static_cast<cl_context>(context);
  const auto native_queue = static_cast<cl_command_queue>(queue);
  OMNI_RETURN_VAL_IF_LOG(clRetainContext(native_context) != CL_SUCCESS, false, ERROR,
                         "Cannot retain OpenCL execution context");
  runtime_->context = native_context;
  OMNI_RETURN_VAL_IF_LOG(clRetainCommandQueue(native_queue) != CL_SUCCESS, false, ERROR,
                         "Cannot retain OpenCL compute queue");
  runtime_->queue = native_queue;
  runtime_->device = static_cast<cl_device_id>(device);
  const char *const profile = std::getenv("OMNI_OPENCL_PROFILE");
  runtime_->enable_profiling = profile != nullptr && std::string(profile) == "1";
  runtime_->root = FindLibraryRoot() / "kernels";
  OMNI_RETURN_VAL_IF(!ReadSource(runtime_->root / "common/opencl_common.cl", &runtime_->prelude),
                     false);
  size_t vendor_bytes = 0U;
  OMNI_RETURN_VAL_IF_LOG(
      clGetDeviceInfo(runtime_->device, CL_DEVICE_VENDOR, 0U, nullptr, &vendor_bytes) != CL_SUCCESS,
      false, ERROR, "Cannot query OpenCL device vendor");
  std::string vendor(vendor_bytes, '\0');
  OMNI_RETURN_VAL_IF_LOG(clGetDeviceInfo(runtime_->device, CL_DEVICE_VENDOR, vendor_bytes,
                                         vendor.data(), nullptr) != CL_SUCCESS,
                         false, ERROR, "Cannot read OpenCL device vendor");
  if (vendor.find("NVIDIA") != std::string::npos) {
    constexpr cl_device_info kComputeCapabilityMajorNv = 0x4000;
    cl_uint major = 0U;
    OMNI_RETURN_VAL_IF_LOG(clGetDeviceInfo(runtime_->device, kComputeCapabilityMajorNv,
                                           sizeof(major), &major, nullptr) != CL_SUCCESS,
                           false, ERROR, "Cannot query NVIDIA OpenCL compute capability");
    runtime_->has_nvidia_warp = major >= 3U;
    runtime_->has_nvidia_mma = major >= 8U;
    if (runtime_->has_nvidia_mma) {
      std::string mma_source;
      OMNI_RETURN_VAL_IF(!ReadSource(runtime_->root / "common/opencl_mma.cl", &mma_source), false);
      runtime_->prelude += "\n" + mma_source;
    }
  }
  cl_device_fp_config fp_config = 0U;
  OMNI_RETURN_VAL_IF_LOG(clGetDeviceInfo(runtime_->device, CL_DEVICE_SINGLE_FP_CONFIG,
                                         sizeof(fp_config), &fp_config, nullptr) != CL_SUCCESS,
                         false, ERROR, "Cannot query OpenCL FP32 arithmetic capabilities");
  runtime_->has_correctly_rounded_division =
      (fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0U;
  OMNI_RETURN_VAL_IF_LOG(clGetDeviceInfo(runtime_->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                         sizeof(runtime_->max_work_group_size),
                                         &runtime_->max_work_group_size, nullptr) != CL_SUCCESS ||
                             clGetDeviceInfo(runtime_->device, CL_DEVICE_LOCAL_MEM_SIZE,
                                             sizeof(runtime_->local_memory_bytes),
                                             &runtime_->local_memory_bytes, nullptr) != CL_SUCCESS,
                         false, ERROR, "Cannot query OpenCL execution limits");
  return true;
}

bool OpenCLExecutor::Launch(const OpenCLInvocation &invocation) {
  std::lock_guard<std::mutex> lock(runtime_->mutex);
  OMNI_RETURN_VAL_IF_LOG(runtime_->context == nullptr || invocation.entry.empty() ||
                             invocation.source.empty() || invocation.dimensions == 0U ||
                             invocation.dimensions > 3U,
                         false, ERROR, "Invalid OpenCL kernel invocation");
  for (uint32_t dimension = 0U; dimension < invocation.dimensions; ++dimension) {
    OMNI_RETURN_VAL_IF_LOG(
        invocation.local_size[dimension] == 0U || invocation.global_size[dimension] == 0U ||
            invocation.global_size[dimension] % invocation.local_size[dimension] != 0U,
        false, ERROR, "Invalid OpenCL work-group dimensions");
  }
  std::string key = invocation.source + ":" + std::to_string(static_cast<int>(invocation.type));
  for (const std::string &define : invocation.build_defines) {
    key += ":" + define;
  }
  auto program_iterator = runtime_->programs.find(key);
  if (program_iterator == runtime_->programs.end()) {
    std::string body;
    OMNI_RETURN_VAL_IF(!ReadSource(runtime_->root / invocation.source, &body), false);
    const std::string source = runtime_->prelude + "\n" + body;
    std::vector<std::string> flags = {
        "-cl-std=CL1.2", "-DOMNI_DTYPE=" + std::to_string(static_cast<int>(invocation.type)),
        "-I\"" + runtime_->root.string() + "\""};
    if (runtime_->has_nvidia_warp) {
      flags.push_back("-DOMNI_NVIDIA_WARP=1");
    }
    if (runtime_->has_nvidia_mma) {
      flags.push_back("-DOMNI_NVIDIA_MMA=1");
    }
    for (const std::string &define : invocation.build_defines) {
      flags.push_back("-D" + define);
    }
    if (runtime_->has_correctly_rounded_division) {
      flags.push_back("-cl-fp32-correctly-rounded-divide-sqrt");
    }
    const opencl::CompilerOptions options = {
        .optimize_level = "strict", .is_debug_enabled = false, .extra_flags = flags};
    std::shared_ptr<opencl::Program> program;
    OMNI_RETURN_VAL_IF(!opencl::Kernel::BuildProgram(runtime_->context, runtime_->device, source,
                                                     options, &program),
                       false);
    program_iterator = runtime_->programs.emplace(key, std::move(program)).first;
  }
  auto prepared = std::make_shared<PreparedKernel>();
  cl_int status = CL_SUCCESS;
  prepared->kernel_ =
      clCreateKernel(program_iterator->second->native_program(), invocation.entry.c_str(), &status);
  OMNI_RETURN_VAL_IF_LOG(status != CL_SUCCESS || prepared->kernel_ == nullptr, false, ERROR,
                         "Cannot create OpenCL kernel " << invocation.entry << ": " << status);
  OMNI_RETURN_VAL_IF(!opencl::Kernel::BindArguments(prepared->kernel_, invocation.arguments),
                     false);
  prepared->global_size_ = invocation.global_size;
  prepared->local_size_ = invocation.local_size;
  prepared->dimensions_ = invocation.dimensions;
  prepared->entry_ = invocation.entry;
  size_t group_size = 1U;
  for (uint32_t dimension = 0U; dimension < invocation.dimensions; ++dimension) {
    OMNI_RETURN_VAL_IF_LOG(
        invocation.local_size[dimension] > runtime_->max_work_group_size / group_size, false, ERROR,
        "OpenCL work-group exceeds device limits: " << invocation.entry);
    group_size *= invocation.local_size[dimension];
  }
  size_t kernel_limit = 0U;
  cl_ulong local_bytes = 0U;
  OMNI_RETURN_VAL_IF_LOG(
      clGetKernelWorkGroupInfo(prepared->kernel_, runtime_->device, CL_KERNEL_WORK_GROUP_SIZE,
                               sizeof(kernel_limit), &kernel_limit, nullptr) != CL_SUCCESS ||
          clGetKernelWorkGroupInfo(prepared->kernel_, runtime_->device, CL_KERNEL_LOCAL_MEM_SIZE,
                                   sizeof(local_bytes), &local_bytes, nullptr) != CL_SUCCESS ||
          group_size > kernel_limit || local_bytes > runtime_->local_memory_bytes,
      false, ERROR, "OpenCL kernel resource limits exceeded: " << invocation.entry);
  if (runtime_->preparation != nullptr) {
    runtime_->preparation->push_back(std::move(prepared));
    return true;
  }
  cl_event event = nullptr;
  status =
      clEnqueueNDRangeKernel(runtime_->queue, prepared->kernel_, prepared->dimensions_, nullptr,
                             prepared->global_size_.data(), prepared->local_size_.data(), 0U,
                             nullptr, runtime_->enable_profiling ? &event : nullptr);
  OMNI_RETURN_VAL_IF_LOG(status != CL_SUCCESS, false, ERROR,
                         "OpenCL enqueue failed: " << invocation.entry << ", status " << status);
  if (event != nullptr) {
    runtime_->samples.emplace_back(invocation.entry, event);
  }
  return true;
}

bool OpenCLExecutor::BeginPreparation(PreparedSequence *const sequence) {
  OMNI_RETURN_VAL_IF_LOG(sequence == nullptr || runtime_->preparation != nullptr, false, ERROR,
                         "Invalid OpenCL preparation state");
  runtime_->preparation = sequence;
  return true;
}

bool OpenCLExecutor::EndPreparation() {
  OMNI_RETURN_VAL_IF_LOG(runtime_->preparation == nullptr, false, ERROR,
                         "No active OpenCL preparation");
  runtime_->preparation = nullptr;
  return true;
}

bool OpenCLExecutor::Enqueue(const PreparedSequence &sequence) {
  for (const auto &prepared : sequence) {
    OMNI_RETURN_VAL_IF_LOG(prepared == nullptr, false, ERROR, "Null prepared OpenCL kernel");
    cl_event event = nullptr;
    const cl_int status =
        clEnqueueNDRangeKernel(runtime_->queue, prepared->kernel_, prepared->dimensions_, nullptr,
                               prepared->global_size_.data(), prepared->local_size_.data(), 0U,
                               nullptr, runtime_->enable_profiling ? &event : nullptr);
    OMNI_RETURN_VAL_IF_LOG(status != CL_SUCCESS, false, ERROR,
                           "Prepared OpenCL enqueue failed: " << status);
    if (event != nullptr) {
      runtime_->samples.emplace_back(prepared->entry_, event);
    }
  }
  return true;
}

bool OpenCLExecutor::HasNvidiaMma() const {
  return runtime_->has_nvidia_mma;
}

} // namespace omni_runtime::backend
