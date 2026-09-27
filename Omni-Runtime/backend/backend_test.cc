#include <cstdint>
#include <array>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "Omni-Runtime/backend/backend.h"

namespace omni_runtime {
namespace backend {
namespace {

void CheckEventTimer(const std::shared_ptr<Backend> &backend) {
  EventTimer timer;
  double elapsed_ms = -1.0;
  EXPECT_FALSE(timer.Stop(&elapsed_ms));
  ASSERT_TRUE(timer.Initialize(backend));
  Allocation allocation;
  ASSERT_TRUE(backend->Allocate(MemoryKind::kDevice, 4U * 1024U * 1024U, &allocation));
  for (int32_t iteration = 0; iteration < 2; ++iteration) {
    ASSERT_TRUE(timer.Start());
    ASSERT_TRUE(backend->Fill(allocation, 1U, nullptr));
    ASSERT_TRUE(timer.Stop(&elapsed_ms));
    EXPECT_GT(elapsed_ms, 0.0);
  }
  EXPECT_FALSE(backend->ElapsedTimeMs(nullptr, nullptr, &elapsed_ms));
  ASSERT_TRUE(backend->Free(&allocation));
}

class CudaBackendTest : public ::testing::Test {
protected:
  void SetUp() override {
    backend_ = CreateBackend(BackendKind::kCuda);
    if (backend_ == nullptr || !backend_->IsAvailable()) {
      GTEST_SKIP() << "CUDA backend is unavailable";
    }
  }

  std::shared_ptr<Backend> backend_;
};

TEST_F(CudaBackendTest, CreatesAndDestroysStream) {
  StreamHandle stream_handle;
  ASSERT_TRUE(backend_->CreateStream(backend::kDefaultStreamFlags, &stream_handle));
  EXPECT_EQ(stream_handle.backend_kind, BackendKind::kCuda);
  ASSERT_TRUE(backend_->SynchronizeStream(stream_handle.native_handle));
  ASSERT_TRUE(backend_->DestroyStream(stream_handle.native_handle));
}

TEST_F(CudaBackendTest, TimesDefaultComputeStreamWithReusableEvents) {
  CheckEventTimer(backend_);
}

TEST_F(CudaBackendTest, SynchronizesStreamsWithEvent) {
  StreamHandle producer_stream;
  StreamHandle consumer_stream;
  EventHandle event_handle;
  ASSERT_TRUE(backend_->CreateStream(kDefaultStreamFlags, &producer_stream));
  ASSERT_TRUE(backend_->CreateStream(kDefaultStreamFlags, &consumer_stream));
  ASSERT_TRUE(backend_->CreateEvent(kDefaultEventFlags, &event_handle));
  ASSERT_TRUE(backend_->RecordEvent(producer_stream.native_handle, &event_handle));
  ASSERT_TRUE(backend_->WaitEvent(consumer_stream.native_handle, event_handle.native_handle,
                                  kDefaultEventWaitFlags));
  ASSERT_TRUE(backend_->SynchronizeStream(consumer_stream.native_handle));
  bool is_event_ready = false;
  ASSERT_TRUE(backend_->QueryEvent(event_handle.native_handle, &is_event_ready));
  EXPECT_TRUE(is_event_ready);
  ASSERT_TRUE(backend_->DestroyEvent(event_handle.native_handle));
  ASSERT_TRUE(backend_->DestroyStream(producer_stream.native_handle));
  ASSERT_TRUE(backend_->DestroyStream(consumer_stream.native_handle));
}

TEST_F(CudaBackendTest, CopiesHostAndDeviceMemory) {
  Allocation allocation;
  ASSERT_TRUE(backend_->Allocate(MemoryKind::kDevice, sizeof(uint32_t) * 4U, &allocation));
  const std::vector<uint32_t> source = {1U, 2U, 3U, 4U};
  std::vector<uint32_t> destination(source.size());
  ASSERT_TRUE(backend_->CopyHostToDevice(source.data(), source.size() * sizeof(uint32_t),
                                         allocation, nullptr));
  ASSERT_TRUE(backend_->CopyDeviceToHost(allocation, destination.size() * sizeof(uint32_t),
                                         destination.data(), nullptr));
  EXPECT_EQ(destination, source);
  ASSERT_TRUE(backend_->Free(&allocation));
}

TEST_F(CudaBackendTest, SupportsPinnedHostAndManagedMemory) {
  EXPECT_TRUE(backend_->IsMemoryKindSupported(MemoryKind::kPinnedHost));
  EXPECT_TRUE(backend_->IsMemoryKindSupported(MemoryKind::kManaged));
  EXPECT_FALSE(backend_->IsMemoryKindSupported(MemoryKind::kHostView));
}

TEST_F(CudaBackendTest, ProjectsRowMajorBatchesWithPersistentCompute) {
  ASSERT_TRUE(backend_->InitializeCompute());
  const size_t compute_workspace_bytes = backend_->compute_workspace_bytes();
  EXPECT_GT(compute_workspace_bytes, 0U);
  ASSERT_TRUE(backend_->InitializeCompute());
  EXPECT_EQ(backend_->compute_workspace_bytes(), compute_workspace_bytes);
  ASSERT_TRUE(backend_->SynchronizeStream(nullptr));
  const std::vector<float> input = {1, 2, 3, 4, -1, 0, 1, 2};
  const std::vector<float> weight = {1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1};
  const std::vector<float> expected = {5, 5, 10, 1, 1, 2};
  std::array<Allocation, 3U> allocations;
  for (Allocation &allocation : allocations) {
    ASSERT_TRUE(
        backend_->Allocate(MemoryKind::kDevice, weight.size() * sizeof(float), &allocation));
  }
  ASSERT_TRUE(backend_->CopyHostToDevice(input.data(), input.size() * sizeof(float), allocations[0],
                                         nullptr));
  ASSERT_TRUE(backend_->CopyHostToDevice(weight.data(), weight.size() * sizeof(float),
                                         allocations[1], nullptr));
  ASSERT_TRUE(backend_->LinearMatmul(ScalarType::kFloat32, allocations[0].data, allocations[1].data,
                                     2, 4, 3, allocations[2].data));
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(backend_->CopyDeviceToHost(allocations[2], actual.size() * sizeof(float),
                                         actual.data(), nullptr));
  EXPECT_EQ(actual, expected);
  ASSERT_TRUE(backend_->LinearMatmul(ScalarType::kFloat32, allocations[0].data, allocations[1].data,
                                     1, 4, 3, allocations[2].data));
  ASSERT_TRUE(
      backend_->CopyDeviceToHost(allocations[2], 3U * sizeof(float), actual.data(), nullptr));
  EXPECT_EQ((std::vector<float>{actual[0], actual[1], actual[2]}), (std::vector<float>{5, 5, 10}));
  ASSERT_TRUE(backend_->Compute(ScalarType::kFloat32, Add{.lhs = allocations[0].data,
                                                          .rhs = allocations[1].data,
                                                          .element_count = 4U,
                                                          .output = allocations[2].data}));
  ASSERT_TRUE(
      backend_->CopyDeviceToHost(allocations[2], 4U * sizeof(float), actual.data(), nullptr));
  EXPECT_EQ((std::vector<float>{actual[0], actual[1], actual[2], actual[3]}),
            (std::vector<float>{2, 2, 3, 5}));
  EXPECT_FALSE(backend_->Compute(ScalarType::kFloat32, Add{.lhs = nullptr,
                                                           .rhs = allocations[1].data,
                                                           .element_count = 4U,
                                                           .output = allocations[2].data}));
  for (Allocation &allocation : allocations) {
    ASSERT_TRUE(backend_->Free(&allocation));
  }
}

class OpenCLBackendTest : public ::testing::Test {
protected:
  void SetUp() override {
    backend_ = CreateBackend(BackendKind::kOpenCL);
    if (backend_ == nullptr || !backend_->IsAvailable()) {
      GTEST_SKIP() << "OpenCL backend is unavailable";
    }
  }

  std::shared_ptr<Backend> backend_;
};

TEST_F(OpenCLBackendTest, CreatesAndDestroysStream) {
  StreamHandle stream_handle;
  ASSERT_TRUE(backend_->CreateStream(backend::kDefaultStreamFlags, &stream_handle));
  EXPECT_EQ(stream_handle.backend_kind, BackendKind::kOpenCL);
  ASSERT_TRUE(backend_->SynchronizeStream(stream_handle.native_handle));
  ASSERT_TRUE(backend_->DestroyStream(stream_handle.native_handle));
}

TEST_F(OpenCLBackendTest, TimesDefaultQueueWithProfilingEvents) {
  CheckEventTimer(backend_);
}

TEST_F(OpenCLBackendTest, CopiesHostAndDeviceMemory) {
  Allocation allocation;
  ASSERT_TRUE(backend_->Allocate(MemoryKind::kDevice, sizeof(uint32_t) * 4U, &allocation));
  StreamHandle stream_handle;
  ASSERT_TRUE(backend_->CreateStream(backend::kDefaultStreamFlags, &stream_handle));
  const std::vector<uint32_t> source = {5U, 6U, 7U, 8U};
  std::vector<uint32_t> destination(source.size());
  ASSERT_TRUE(backend_->CopyHostToDevice(source.data(), source.size() * sizeof(uint32_t),
                                         allocation, stream_handle.native_handle));
  ASSERT_TRUE(backend_->CopyDeviceToHost(allocation, destination.size() * sizeof(uint32_t),
                                         destination.data(), stream_handle.native_handle));
  ASSERT_TRUE(backend_->SynchronizeStream(stream_handle.native_handle));
  EXPECT_EQ(destination, source);
  ASSERT_TRUE(backend_->DestroyStream(stream_handle.native_handle));
  ASSERT_TRUE(backend_->Free(&allocation));
}

TEST_F(OpenCLBackendTest, CopiesHostAndDeviceMemorySynchronously) {
  Allocation allocation;
  ASSERT_TRUE(backend_->Allocate(MemoryKind::kDevice, sizeof(uint32_t) * 4U, &allocation));
  const std::vector<uint32_t> source = {9U, 10U, 11U, 12U};
  std::vector<uint32_t> destination(source.size());
  ASSERT_TRUE(backend_->CopyHostToDevice(source.data(), source.size() * sizeof(uint32_t),
                                         allocation, nullptr));
  ASSERT_TRUE(backend_->CopyDeviceToHost(allocation, destination.size() * sizeof(uint32_t),
                                         destination.data(), nullptr));
  EXPECT_EQ(destination, source);
  ASSERT_TRUE(backend_->Free(&allocation));
}

TEST_F(OpenCLBackendTest, RejectsUnsupportedMemoryKind) {
  EXPECT_TRUE(backend_->IsMemoryKindSupported(MemoryKind::kDevice));
  EXPECT_FALSE(backend_->IsMemoryKindSupported(MemoryKind::kManaged));
}

TEST_F(OpenCLBackendTest, RejectsUnsupportedInferenceComputeExplicitly) {
  EXPECT_FALSE(backend_->InitializeCompute());
  EXPECT_FALSE(backend_->Compute(
      ScalarType::kFloat16,
      Add{.lhs = nullptr, .rhs = nullptr, .element_count = 1U, .output = nullptr}));
}

} // namespace
} // namespace backend
} // namespace omni_runtime
