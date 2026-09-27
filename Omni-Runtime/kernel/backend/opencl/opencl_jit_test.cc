#define CL_TARGET_OPENCL_VERSION 300

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <CL/cl.h>
#include <gtest/gtest.h>

#include "Omni-Runtime/kernel/backend/opencl/backend.h"
#include "Omni-Runtime/kernel/backend/opencl/kernel.h"
#include "Omni-Runtime/utils/logger.h"

namespace omni_runtime::jit::backend::opencl {
namespace {

constexpr std::size_t kQwen3VlHiddenSize = 4096U;
constexpr std::size_t kWorkGroupSize = 64U;

struct ClMemDeleter final {
  void operator()(cl_mem memory) const {
    if (memory != nullptr) {
      static_cast<void>(clReleaseMemObject(memory));
    }
  }
};

struct ClQueueDeleter final {
  void operator()(cl_command_queue queue) const {
    if (queue != nullptr) {
      static_cast<void>(clReleaseCommandQueue(queue));
    }
  }
};

using ClMemPtr = std::unique_ptr<std::remove_pointer_t<cl_mem>, ClMemDeleter>;
using ClQueuePtr = std::unique_ptr<std::remove_pointer_t<cl_command_queue>, ClQueueDeleter>;

class OpenClJitTest : public ::testing::Test {
protected:
  void SetUp() override {
    backend_ = std::make_unique<OpenCL>();
    if (!backend_->IsAvailable()) {
      GTEST_SKIP() << "OpenCL JIT backend is unavailable";
    }
  }

  bool CreateMemory(const cl_mem_flags flags, const std::size_t size_bytes,
                    ClMemPtr *const memory) const {
    OMNI_RETURN_VAL_IF(memory == nullptr, false);
    cl_int status = CL_SUCCESS;
    cl_mem native_memory = clCreateBuffer(backend_->context(), flags, size_bytes, nullptr, &status);
    OMNI_RETURN_VAL_IF(status != CL_SUCCESS || native_memory == nullptr, false);
    memory->reset(native_memory);
    return true;
  }

  ClQueuePtr CreateQueue() const {
    cl_int status = CL_SUCCESS;
    cl_command_queue native_queue = clCreateCommandQueueWithProperties(
        backend_->context(), backend_->device(), nullptr, &status);
    OMNI_RETURN_VAL_IF(status != CL_SUCCESS || native_queue == nullptr, ClQueuePtr());
    return ClQueuePtr(native_queue);
  }

  std::unique_ptr<OpenCL> backend_;
};

TEST_F(OpenClJitTest, CompilesAndLaunchesVectorAddKernel) {
  std::vector<float> lhs(kQwen3VlHiddenSize);
  std::vector<float> rhs(kQwen3VlHiddenSize);
  std::vector<float> expected(kQwen3VlHiddenSize);
  for (std::size_t index = 0U; index < kQwen3VlHiddenSize; ++index) {
    lhs[index] = 0.01F * static_cast<float>(index % 128U);
    rhs[index] = 0.001F * static_cast<float>(index);
    expected[index] = lhs[index] + rhs[index];
  }

  const std::string source = R"OpenCL(
__kernel void vector_add(__global const float* lhs, __global const float* rhs,
                         __global float* output) {
  const size_t index = get_global_id(0);
  output[index] = lhs[index] + rhs[index];
}
)OpenCL";
  const std::filesystem::path source_path =
      std::filesystem::temp_directory_path() / "omni_opencl_jit_test" / "vector_add.cl";
  std::shared_ptr<Kernel> kernel;
  ASSERT_TRUE(backend_->Compile(source, source_path, "vector_add",
                                backend_->default_compiler_options(), &kernel));
  ASSERT_NE(kernel, nullptr);

  const ClQueuePtr queue = CreateQueue();
  ASSERT_NE(queue, nullptr);
  ClMemPtr lhs_memory;
  ClMemPtr rhs_memory;
  ClMemPtr output_memory;
  const std::size_t size_bytes = lhs.size() * sizeof(float);
  ASSERT_TRUE(CreateMemory(CL_MEM_READ_ONLY, size_bytes, &lhs_memory));
  ASSERT_TRUE(CreateMemory(CL_MEM_READ_ONLY, size_bytes, &rhs_memory));
  ASSERT_TRUE(CreateMemory(CL_MEM_WRITE_ONLY, size_bytes, &output_memory));
  ASSERT_EQ(clEnqueueWriteBuffer(queue.get(), lhs_memory.get(), CL_TRUE, 0U, size_bytes, lhs.data(),
                                 0U, nullptr, nullptr),
            CL_SUCCESS);
  ASSERT_EQ(clEnqueueWriteBuffer(queue.get(), rhs_memory.get(), CL_TRUE, 0U, size_bytes, rhs.data(),
                                 0U, nullptr, nullptr),
            CL_SUCCESS);

  const std::vector<const void *> arguments = {lhs_memory.get(), rhs_memory.get(),
                                               output_memory.get()};
  const std::vector<std::size_t> global_size = {kQwen3VlHiddenSize};
  const std::vector<std::size_t> local_size = {kWorkGroupSize};
  ASSERT_TRUE(kernel->Launch(queue.get(), global_size, local_size, arguments));
  ASSERT_EQ(clFinish(queue.get()), CL_SUCCESS);

  std::vector<float> actual(expected.size());
  ASSERT_EQ(clEnqueueReadBuffer(queue.get(), output_memory.get(), CL_TRUE, 0U, size_bytes,
                                actual.data(), 0U, nullptr, nullptr),
            CL_SUCCESS);
  EXPECT_EQ(actual, expected);
}

} // namespace
} // namespace omni_runtime::jit::backend::opencl
