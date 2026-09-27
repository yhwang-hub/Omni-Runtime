#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "Omni-Runtime/backend/backend.h"

namespace omni_runtime {
namespace backend {
namespace {

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

} // namespace
} // namespace backend
} // namespace omni_runtime
