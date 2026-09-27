#include "Omni-Runtime/backend/backend_opencl.h"

#include <algorithm>
#include <vector>

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
    runtime_->default_queue =
        clCreateCommandQueueWithProperties(runtime_->context, runtime_->device, nullptr, nullptr);
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
  const cl_int status = clReleaseMemObject(NativeMemory(*allocation));
  allocation->data = nullptr;
  allocation->size_bytes = 0U;
  return status == CL_SUCCESS;
}

bool OpenCLBackend::Fill(const Allocation &allocation, const uint8_t value, const void *stream) {
  OMNI_RETURN_VAL_IF(!IsAvailable() || allocation.data == nullptr, false);
  const cl_uchar pattern = value;
  return Submit(stream, [&](cl_command_queue queue) {
    return clEnqueueFillBuffer(queue, NativeMemory(allocation), &pattern, sizeof(pattern), 0U,
                               allocation.size_bytes, 0U, nullptr, nullptr);
  });
}

bool OpenCLBackend::CopyHostToDevice(const void *const source, const std::size_t size_bytes,
                                     const Allocation &destination, const void *stream) {
  OMNI_RETURN_VAL_IF(!IsAvailable() || source == nullptr || destination.data == nullptr, false);
  return Submit(stream, [&](cl_command_queue queue) {
    return clEnqueueWriteBuffer(queue, NativeMemory(destination), CL_FALSE, 0U, size_bytes, source,
                                0U, nullptr, nullptr);
  });
}

bool OpenCLBackend::CopyDeviceToHost(const Allocation &source, const std::size_t size_bytes,
                                     void *const destination, const void *stream) {
  OMNI_RETURN_VAL_IF(!IsAvailable() || destination == nullptr || source.data == nullptr, false);
  return Submit(stream, [&](cl_command_queue queue) {
    return clEnqueueReadBuffer(queue, NativeMemory(source), CL_FALSE, 0U, size_bytes, destination,
                               0U, nullptr, nullptr);
  });
}

bool OpenCLBackend::CopyDeviceToDevice(const Allocation &source, const Allocation &destination,
                                       const std::size_t size_bytes, const void *stream) {
  OMNI_RETURN_VAL_IF(!IsAvailable() || source.data == nullptr || destination.data == nullptr, false);
  return Submit(stream, [&](cl_command_queue queue) {
    return clEnqueueCopyBuffer(queue, NativeMemory(source), NativeMemory(destination), 0U, 0U,
                               size_bytes, 0U, nullptr, nullptr);
  });
}

bool OpenCLBackend::Submit(const void *stream,
                           const std::function<cl_int(cl_command_queue)> &operation) {
  cl_command_queue queue = stream == nullptr ? runtime_->default_queue : NativeStream(stream);
  OMNI_RETURN_VAL_IF(queue == nullptr, false);
  if (operation(queue) != CL_SUCCESS) {
    return false;
  }
  return stream != nullptr || clFinish(queue) == CL_SUCCESS;
}

bool OpenCLBackend::CreateStream(const uint32_t flags, StreamHandle *const stream_handle) {
  static_cast<void>(flags);
  OMNI_RETURN_VAL_IF(!IsAvailable() || stream_handle == nullptr, false);
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(runtime_->context, runtime_->device, nullptr, nullptr);
  OMNI_RETURN_VAL_IF(queue == nullptr, false);
  stream_handle->native_handle = queue;
  stream_handle->backend_kind = BackendKind::kOpenCL;
  return true;
}

bool OpenCLBackend::SynchronizeStream(const void *stream) {
  OMNI_RETURN_VAL_IF(stream == nullptr, false);
  return clFinish(NativeStream(stream)) == CL_SUCCESS;
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
  OMNI_RETURN_VAL_IF(event_handle == nullptr || stream == nullptr, false);
  cl_event event = nullptr;
  OMNI_RETURN_VAL_IF(
      clEnqueueMarkerWithWaitList(NativeStream(stream), 0U, nullptr, &event) != CL_SUCCESS, false);
  if (event_handle->native_handle != nullptr) {
    static_cast<void>(clReleaseEvent(reinterpret_cast<cl_event>(event_handle->native_handle)));
  }
  event_handle->native_handle = event;
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
