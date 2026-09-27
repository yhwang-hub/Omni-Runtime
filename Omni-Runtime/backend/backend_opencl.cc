#include "Omni-Runtime/backend/backend_opencl.h"

#include <algorithm>
#include <vector>

#include "Omni-Runtime/backend/opencl_jit/executor.h"
#include "Omni-Runtime/kernels/compute_runtime.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime {
namespace backend {
namespace {

cl_command_queue NativeStream(const void *stream) {
  return reinterpret_cast<cl_command_queue>(const_cast<void *>(stream));
}

cl_mem NativeMemory(const Allocation &allocation) {
  return reinterpret_cast<cl_mem>(allocation.data);
}

} // namespace

struct OpenCLBackend::Runtime final {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue default_queue = nullptr;
  std::shared_ptr<OpenCLExecutor> executor;
};

OpenCLBackend::OpenCLBackend() : runtime_(std::make_unique<Runtime>()) {
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
    const cl_int status = clRetainDevice(devices.front());
    if (status != CL_SUCCESS) {
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
      if (clRetainDevice(devices.front()) != CL_SUCCESS) {
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
  if (runtime_->context != nullptr) {
    const cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    runtime_->default_queue = clCreateCommandQueueWithProperties(
        runtime_->context, runtime_->device, properties, nullptr);
  }
}

OpenCLBackend::~OpenCLBackend() {
  if (runtime_ == nullptr) {
    return;
  }
  if (runtime_->default_queue != nullptr) {
    clReleaseCommandQueue(runtime_->default_queue);
  }
  if (runtime_->context != nullptr) {
    clReleaseContext(runtime_->context);
  }
  if (runtime_->device != nullptr) {
    clReleaseDevice(runtime_->device);
  }
}

bool OpenCLBackend::IsAvailable() const {
  return runtime_ != nullptr && runtime_->context != nullptr && runtime_->default_queue != nullptr;
}

BackendKind OpenCLBackend::kind() const {
  return BackendKind::kOpenCL;
}

bool OpenCLBackend::IsMemoryKindSupported(const MemoryKind memory_kind) const {
  return memory_kind == MemoryKind::kDevice;
}

bool OpenCLBackend::Allocate(const MemoryKind memory_kind, const std::size_t size_bytes,
                             Allocation *const allocation) {
  OMNI_RETURN_VAL_IF(!IsAvailable() || allocation == nullptr || size_bytes == 0U, false);
  OMNI_RETURN_VAL_IF(!IsMemoryKindSupported(memory_kind), false);
  OMNI_RETURN_VAL_IF(allocation->data != nullptr, false);

  cl_int status = CL_SUCCESS;
  const cl_mem memory =
      clCreateBuffer(runtime_->context, CL_MEM_READ_WRITE, size_bytes, nullptr, &status);
  OMNI_RETURN_VAL_IF(status != CL_SUCCESS || memory == nullptr, false);

  allocation->data = memory;
  allocation->size_bytes = size_bytes;
  allocation->memory_kind = memory_kind;
  return true;
}

bool OpenCLBackend::Free(Allocation *const allocation) {
  OMNI_RETURN_VAL_IF(allocation == nullptr, false);
  OMNI_RETURN_VAL_IF(allocation->data == nullptr, true);
  OMNI_RETURN_VAL_IF_LOG(allocation->offset_bytes != 0U ||
                             allocation->memory_kind == MemoryKind::kDeviceView,
                         false, ERROR, "Cannot free a non-owning device view");
  const cl_int status = clReleaseMemObject(NativeMemory(*allocation));
  allocation->data = nullptr;
  allocation->size_bytes = 0U;
  allocation->offset_bytes = 0U;
  return status == CL_SUCCESS;
}

bool OpenCLBackend::Fill(const Allocation &allocation, const uint8_t value, const void *stream) {
  OMNI_RETURN_VAL_IF(!IsAvailable() || allocation.data == nullptr, false);
  const cl_uchar pattern = value;
  return Submit(stream, [&](cl_command_queue queue) {
    return clEnqueueFillBuffer(queue, NativeMemory(allocation), &pattern, sizeof(pattern),
                               allocation.offset_bytes, allocation.size_bytes, 0U, nullptr,
                               nullptr);
  });
}

bool OpenCLBackend::CopyHostToDevice(const void *const source, const std::size_t size_bytes,
                                     const Allocation &destination, const void *stream) {
  OMNI_RETURN_VAL_IF(!IsAvailable() || source == nullptr || destination.data == nullptr, false);
  OMNI_RETURN_VAL_IF_LOG(size_bytes > destination.size_bytes, false, ERROR,
                         "Host upload exceeds the destination view");
  return Submit(stream, [&](cl_command_queue queue) {
    return clEnqueueWriteBuffer(queue, NativeMemory(destination), CL_FALSE,
                                destination.offset_bytes, size_bytes, source, 0U, nullptr, nullptr);
  });
}

bool OpenCLBackend::CopyDeviceToHost(const Allocation &source, const std::size_t size_bytes,
                                     void *const destination, const void *stream) {
  OMNI_RETURN_VAL_IF(!IsAvailable() || destination == nullptr || source.data == nullptr, false);
  OMNI_RETURN_VAL_IF_LOG(size_bytes > source.size_bytes, false, ERROR,
                         "Device readback exceeds the source view");
  return Submit(stream, [&](cl_command_queue queue) {
    return clEnqueueReadBuffer(queue, NativeMemory(source), CL_FALSE, source.offset_bytes,
                               size_bytes, destination, 0U, nullptr, nullptr);
  });
}

bool OpenCLBackend::CopyDeviceToDevice(const Allocation &source, const Allocation &destination,
                                       const std::size_t size_bytes, const void *stream) {
  OMNI_RETURN_VAL_IF(!IsAvailable() || source.data == nullptr || destination.data == nullptr,
                     false);
  OMNI_RETURN_VAL_IF_LOG(size_bytes > source.size_bytes || size_bytes > destination.size_bytes,
                         false, ERROR, "Device copy exceeds a buffer view");
  return Submit(stream, [&](cl_command_queue queue) {
    return clEnqueueCopyBuffer(queue, NativeMemory(source), NativeMemory(destination),
                               source.offset_bytes, destination.offset_bytes, size_bytes, 0U,
                               nullptr, nullptr);
  });
}

bool OpenCLBackend::Submit(const void *stream,
                           const std::function<cl_int(cl_command_queue)> &operation) {
  cl_command_queue queue = stream == nullptr ? runtime_->default_queue : NativeStream(stream);
  OMNI_RETURN_VAL_IF(queue == nullptr, false);
  const cl_int status = operation(queue);
  OMNI_RETURN_VAL_IF_LOG(status != CL_SUCCESS, false, ERROR,
                         "OpenCL memory submission failed: " << status);
  OMNI_RETURN_VAL_IF_LOG(stream == nullptr && clFinish(queue) != CL_SUCCESS, false, ERROR,
                         "OpenCL memory synchronization failed");
  return true;
}

bool OpenCLBackend::InitializeCompute() {
  OMNI_RETURN_VAL_IF_LOG(!IsAvailable(), false, ERROR, "OpenCL backend is unavailable");
  OMNI_RETURN_VAL_IF(runtime_->executor != nullptr, true);
  auto executor = std::make_shared<OpenCLExecutor>();
  OMNI_RETURN_VAL_IF(
      !executor->Initialize(runtime_->context, runtime_->device, runtime_->default_queue), false);
  runtime_->executor = std::move(executor);
  LOG_INFO("OpenCL compute uses FP32 accumulation with explicit FP16/BF16 storage rounding");
  return true;
}

bool OpenCLBackend::Compute(const ScalarType type, const ComputeOperation &operation) {
  OMNI_RETURN_VAL_IF_LOG(runtime_->executor == nullptr, false, ERROR,
                         "OpenCL compute backend is not initialized");
  const auto launch = [&]<typename T>() {
    return std::visit(
        [&](const auto &args) {
          if constexpr (std::is_same_v<std::decay_t<decltype(args)>, ParallelLinear>) {
            OMNI_RETURN_VAL_IF(!ValidateParallelLinear(type, kind(), args), false);
            bool is_enabled = false;
            OMNI_RETURN_VAL_IF(!GetParallelExecutionEnabled(&is_enabled), false);
            if (!is_enabled) {
              for (const Linear &projection : args.projections) {
                OMNI_RETURN_VAL_IF(!Compute(type, ComputeOperation(projection)), false);
              }
              return true;
            }
            if (args.projections.front().rows > 1) {
              std::vector<OpenCLInvocation> branches;
              for (const Linear &projection : args.projections) {
                OpenCLInvocation branch;
                branch.type = type;
                branch.has_nvidia_mma = runtime_->executor->HasNvidiaMma();
                branch.compute_unit_count = runtime_->executor->compute_unit_count();
                OMNI_RETURN_VAL_IF(
                    !kernels::runtime::OpenCLOperator<Linear>::Make(projection, &branch), false);
                branches.push_back(std::move(branch));
              }
              const auto &first = branches.front();
              const bool can_combine =
                  std::all_of(branches.begin(), branches.end(), [&first](const auto &branch) {
                    return branch.entry == first.entry &&
                           branch.build_defines == first.build_defines &&
                           branch.local_size == first.local_size;
                  });
              if (!can_combine) {
                for (const auto &branch : branches) {
                  OMNI_RETURN_VAL_IF(!runtime_->executor->Launch(branch), false);
                }
                return true;
              }
            }
            OpenCLInvocation invocation;
            invocation.type = type;
            invocation.has_nvidia_mma = runtime_->executor->HasNvidiaMma();
            invocation.compute_unit_count = runtime_->executor->compute_unit_count();
            OMNI_RETURN_VAL_IF(
                !kernels::runtime::OpenCLOperator<ParallelLinear>::Make(args, &invocation), false);
            return runtime_->executor->Launch(invocation);
          } else {
            return kernels::runtime::OperatorRuntime<T, std::decay_t<decltype(args)>>::Launch(
                BackendKind::kOpenCL, args, runtime_->executor.get());
          }
        },
        operation);
  };
  bool is_successful = false;
  switch (type) {
  case ScalarType::kFloat32:
    is_successful = launch.template operator()<float>();
    break;
  case ScalarType::kFloat16:
    is_successful = launch.template operator()<kernels::Float16>();
    break;
  case ScalarType::kBFloat16:
    is_successful = launch.template operator()<kernels::BFloat16>();
    break;
  }
  OMNI_RETURN_VAL_IF_LOG(!is_successful, false, ERROR, "OpenCL compute operation failed");
  return true;
}

bool OpenCLBackend::LinearMatmul(const ScalarType type, const DeviceView &input,
                                 const DeviceView &weight, const int32_t rows, const int32_t inner,
                                 const int32_t columns, const DeviceView &output) {
  return Compute(type, Linear{.input = input,
                              .weight = weight,
                              .rows = rows,
                              .inner = inner,
                              .columns = columns,
                              .output = output});
}

class OpenCLBackend::OpenCLExecutionPlan final : public ExecutionPlan {
public:
  ~OpenCLExecutionPlan() override {
    for (const cl_event event : events_) {
      if (event != nullptr) {
        static_cast<void>(clReleaseEvent(event));
      }
    }
  }

  bool Initialize(OpenCLBackend *const backend, const ScalarType type,
                  const std::vector<ComputeOperation> &operations, const Allocation &readback) {
    OMNI_RETURN_VAL_IF(backend == nullptr, false);
    executor_ = backend->runtime_->executor;
    queue_ = backend->runtime_->default_queue;
    readback_ = readback;
    OMNI_RETURN_VAL_IF(!executor_->BeginPreparation(&kernels_), false);
    bool is_prepared = true;
    for (const ComputeOperation &operation : operations) {
      if (!backend->Compute(type, operation)) {
        is_prepared = false;
        break;
      }
    }
    const bool is_finished = executor_->EndPreparation();
    return is_prepared && is_finished;
  }

  bool Run(const size_t result_bytes, void *const result, double *const elapsed_ms) override {
    OMNI_RETURN_VAL_IF_LOG(result == nullptr || elapsed_ms == nullptr ||
                               result_bytes != readback_.size_bytes || kernels_.empty() ||
                               !is_usable_,
                           false, ERROR, "Invalid execution plan outputs or state");
    if (!Execute(result_bytes, result, elapsed_ms)) {
      is_usable_ = false;
      if (clFinish(queue_) != CL_SUCCESS) {
        LOG_ERROR("Failed to drain an invalid OpenCL execution plan");
      }
      return false;
    }
    return true;
  }

private:
  bool Execute(const size_t result_bytes, void *const result, double *const elapsed_ms) {
    for (cl_event &event : events_) {
      if (event != nullptr) {
        static_cast<void>(clReleaseEvent(event));
        event = nullptr;
      }
    }
    OMNI_RETURN_VAL_IF_LOG(
        clEnqueueMarkerWithWaitList(queue_, 0U, nullptr, &events_[0]) != CL_SUCCESS ||
            !executor_->Enqueue(kernels_) ||
            clEnqueueMarkerWithWaitList(queue_, 0U, nullptr, &events_[1]) != CL_SUCCESS ||
            clEnqueueReadBuffer(queue_, NativeMemory(readback_), CL_FALSE, readback_.offset_bytes,
                                result_bytes, result, 0U, nullptr, &events_[2]) != CL_SUCCESS ||
            clWaitForEvents(1U, &events_[2]) != CL_SUCCESS,
        false, ERROR, "OpenCL execution or result readback failed");
    cl_ulong start_ns = 0U;
    cl_ulong end_ns = 0U;
    OMNI_RETURN_VAL_IF_LOG(
        clGetEventProfilingInfo(events_[0], CL_PROFILING_COMMAND_END, sizeof(start_ns), &start_ns,
                                nullptr) != CL_SUCCESS ||
            clGetEventProfilingInfo(events_[1], CL_PROFILING_COMMAND_END, sizeof(end_ns), &end_ns,
                                    nullptr) != CL_SUCCESS ||
            end_ns < start_ns,
        false, ERROR, "OpenCL execution timing failed");
    *elapsed_ms = static_cast<double>(end_ns - start_ns) / 1000000.0;
    return true;
  }

  bool is_usable_ = true;
  std::shared_ptr<OpenCLExecutor> executor_;
  OpenCLExecutor::PreparedSequence kernels_;
  cl_command_queue queue_ = nullptr;
  Allocation readback_;
  std::array<cl_event, 3U> events_ = {};
};

bool OpenCLBackend::CreateExecutionPlan(const ScalarType type,
                                        const std::vector<ComputeOperation> &operations,
                                        const Allocation &readback,
                                        std::unique_ptr<ExecutionPlan> *const plan) {
  OMNI_RETURN_VAL_IF_LOG(runtime_->executor == nullptr || plan == nullptr || operations.empty() ||
                             readback.data == nullptr || readback.size_bytes == 0U ||
                             (readback.memory_kind != MemoryKind::kDevice &&
                              readback.memory_kind != MemoryKind::kDeviceView),
                         false, ERROR, "Invalid OpenCL execution plan inputs");
  auto prepared = std::make_unique<OpenCLExecutionPlan>();
  OMNI_RETURN_VAL_IF(!prepared->Initialize(this, type, operations, readback), false);
  *plan = std::move(prepared);
  return true;
}

bool OpenCLBackend::CreateStream(const uint32_t flags, StreamHandle *const stream_handle) {
  static_cast<void>(flags);
  OMNI_RETURN_VAL_IF(!IsAvailable() || stream_handle == nullptr, false);
  const cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(runtime_->context, runtime_->device, properties, nullptr);
  OMNI_RETURN_VAL_IF(queue == nullptr, false);
  stream_handle->native_handle = queue;
  stream_handle->backend_kind = BackendKind::kOpenCL;
  return true;
}

bool OpenCLBackend::SynchronizeStream(const void *stream) {
  OMNI_RETURN_VAL_IF(!IsAvailable(), false);
  return clFinish(stream == nullptr ? runtime_->default_queue : NativeStream(stream)) == CL_SUCCESS;
}

bool OpenCLBackend::QueryStream(const void *stream, bool *const is_ready) {
  OMNI_RETURN_VAL_IF(stream == nullptr || is_ready == nullptr, false);
  cl_event marker = nullptr;
  if (clEnqueueMarkerWithWaitList(NativeStream(stream), 0U, nullptr, &marker) != CL_SUCCESS) {
    return false;
  }
  static_cast<void>(clFlush(NativeStream(stream)));
  cl_int status = CL_COMPLETE;
  const bool is_query_success = clGetEventInfo(marker, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                               sizeof(status), &status, nullptr) == CL_SUCCESS;
  clReleaseEvent(marker);
  if (!is_query_success) {
    return false;
  }
  *is_ready = status == CL_COMPLETE;
  return true;
}

bool OpenCLBackend::WaitEvent(const void *stream, const void *event, const uint32_t flags) {
  static_cast<void>(flags);
  OMNI_RETURN_VAL_IF(stream == nullptr || event == nullptr, false);
  cl_event wait_event = reinterpret_cast<cl_event>(const_cast<void *>(event));
  return clEnqueueBarrierWithWaitList(NativeStream(stream), 1U, &wait_event, nullptr) == CL_SUCCESS;
}

bool OpenCLBackend::CreateEvent(const uint32_t flags, EventHandle *const event_handle) {
  static_cast<void>(flags);
  OMNI_RETURN_VAL_IF(event_handle == nullptr || event_handle->native_handle != nullptr, false);
  event_handle->native_handle = nullptr;
  event_handle->backend_kind = BackendKind::kOpenCL;
  return true;
}

bool OpenCLBackend::DestroyEvent(void *event) {
  if (event == nullptr) {
    return true;
  }
  return clReleaseEvent(reinterpret_cast<cl_event>(event)) == CL_SUCCESS;
}

bool OpenCLBackend::RecordEvent(const void *stream, EventHandle *const event_handle) {
  OMNI_RETURN_VAL_IF(!IsAvailable() || event_handle == nullptr, false);
  cl_command_queue queue = stream == nullptr ? runtime_->default_queue : NativeStream(stream);
  cl_event event = nullptr;
  OMNI_RETURN_VAL_IF(clEnqueueMarkerWithWaitList(queue, 0U, nullptr, &event) != CL_SUCCESS, false);
  if (event_handle->native_handle != nullptr) {
    static_cast<void>(clReleaseEvent(reinterpret_cast<cl_event>(event_handle->native_handle)));
  }
  event_handle->native_handle = event;
  return clFlush(queue) == CL_SUCCESS;
}

bool OpenCLBackend::ElapsedTimeMs(const void *const start, const void *const end,
                                  double *const elapsed_ms) {
  OMNI_RETURN_VAL_IF(start == nullptr || end == nullptr || elapsed_ms == nullptr, false);
  cl_ulong start_ns = 0;
  cl_ulong end_ns = 0;
  OMNI_RETURN_VAL_IF_LOG(
      clGetEventProfilingInfo(reinterpret_cast<cl_event>(const_cast<void *>(start)),
                              CL_PROFILING_COMMAND_END, sizeof(start_ns), &start_ns,
                              nullptr) != CL_SUCCESS ||
          clGetEventProfilingInfo(reinterpret_cast<cl_event>(const_cast<void *>(end)),
                                  CL_PROFILING_COMMAND_END, sizeof(end_ns), &end_ns,
                                  nullptr) != CL_SUCCESS ||
          end_ns < start_ns,
      false, ERROR, "OpenCL event profiling is unavailable or timestamps are invalid");
  *elapsed_ms = static_cast<double>(end_ns - start_ns) / 1000000.0;
  return true;
}

bool OpenCLBackend::SynchronizeEvent(const void *event) {
  OMNI_RETURN_VAL_IF(event == nullptr, false);
  cl_event native_event = reinterpret_cast<cl_event>(const_cast<void *>(event));
  return clWaitForEvents(1U, &native_event) == CL_SUCCESS;
}

bool OpenCLBackend::QueryEvent(const void *event, bool *const is_ready) {
  OMNI_RETURN_VAL_IF(event == nullptr || is_ready == nullptr, false);
  *is_ready = false;
  cl_int status = CL_COMPLETE;
  OMNI_RETURN_VAL_IF(clGetEventInfo(reinterpret_cast<cl_event>(const_cast<void *>(event)),
                                    CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status,
                                    nullptr) != CL_SUCCESS,
                     false);
  *is_ready = status == CL_COMPLETE;
  return true;
}

bool OpenCLBackend::DestroyStream(void *stream) {
  if (stream == nullptr) {
    return true;
  }
  return clReleaseCommandQueue(reinterpret_cast<cl_command_queue>(stream)) == CL_SUCCESS;
}

} // namespace backend
} // namespace omni_runtime
