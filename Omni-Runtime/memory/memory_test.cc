#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "Omni-Runtime/backend/backend.h"
#include "Omni-Runtime/memory/backend_event.h"
#include "Omni-Runtime/memory/backend_stream.h"
#include "Omni-Runtime/memory/buffer.h"
#include "Omni-Runtime/memory/tensor.h"

namespace omni_runtime {
namespace memory {
namespace {

class MemoryTest : public ::testing::Test {
protected:
  void SetUp() override {
    backend_ = backend::CreateBackend(backend::BackendKind::kCuda);
    if (backend_ == nullptr || !backend_->IsAvailable()) {
      GTEST_SKIP() << "CUDA backend is unavailable";
    }
  }

  std::shared_ptr<backend::Backend> backend_;
};

TEST_F(MemoryTest, HostBufferRoundTrip) {
  const BufferPtr buffer = Buffer::CreateOwnedBuffer(nullptr, 8U, backend::MemoryKind::kHost);
  ASSERT_NE(buffer, nullptr);
  const std::vector<uint32_t> source = {1U, 2U};
  std::vector<uint32_t> destination;
  ASSERT_TRUE(buffer->CopyDataFromVector(source));
  ASSERT_TRUE(buffer->CopyDataToVector(&destination));
  EXPECT_EQ(destination, source);
}

TEST_F(MemoryTest, DeviceBufferRoundTrip) {
  const BufferPtr buffer = Buffer::CreateOwnedBuffer(backend_, 8U, backend::MemoryKind::kDevice);
  ASSERT_NE(buffer, nullptr);
  const std::vector<uint32_t> source = {3U, 4U};
  std::vector<uint32_t> destination;
  ASSERT_TRUE(buffer->CopyDataFromVector(source));
  ASSERT_TRUE(buffer->CopyDataToVector(&destination));
  EXPECT_EQ(destination, source);
}

TEST_F(MemoryTest, TypedDeviceBufferRoundTrip) {
  const std::vector<uint32_t> source = {25U, 26U};
  const BufferPtr buffer = Buffer::CreateDeviceBufferFromVector(backend_, source);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->element_count<uint32_t>(), source.size());
  EXPECT_NE(buffer->device_address(), nullptr);
  std::vector<uint32_t> destination;
  ASSERT_TRUE(buffer->CopyDataToVector(&destination));
  EXPECT_EQ(destination, source);
}

TEST_F(MemoryTest, PinnedHostBufferRoundTrip) {
  const BufferPtr buffer =
      Buffer::CreateOwnedBuffer(backend_, 16U, backend::MemoryKind::kPinnedHost);
  ASSERT_NE(buffer, nullptr);
  const std::vector<uint8_t> source = {5U, 6U, 7U, 8U};
  std::vector<uint8_t> destination(source.size());
  ASSERT_TRUE(buffer->CopyDataFromVector(source));
  ASSERT_TRUE(buffer->CopyToHost(source.size(), destination.data()));
  EXPECT_EQ(destination, source);
  EXPECT_TRUE(buffer->is_host_accessible());
}

TEST_F(MemoryTest, ManagedBufferRoundTrip) {
  const BufferPtr buffer = Buffer::CreateOwnedBuffer(backend_, 12U, backend::MemoryKind::kManaged);
  ASSERT_NE(buffer, nullptr);
  const std::vector<uint32_t> source = {9U, 10U, 11U};
  std::vector<uint32_t> destination;
  ASSERT_TRUE(buffer->CopyDataFromVector(source));
  ASSERT_TRUE(buffer->CopyDataToVector(&destination));
  EXPECT_EQ(destination, source);
  EXPECT_TRUE(buffer->is_host_accessible());
}

TEST_F(MemoryTest, HostViewTracksExternalMemory) {
  std::vector<uint8_t> source = {12U, 13U, 14U};
  const BufferPtr buffer = Buffer::CreateHostBufferView(source.data(), source.size());
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->memory_kind(), backend::MemoryKind::kHostView);
  EXPECT_EQ(buffer->size_bytes(), source.size());
  EXPECT_EQ(buffer->host_address(), source.data());
  source[0] = 15U;
  EXPECT_EQ(static_cast<uint8_t *>(buffer->host_address())[0], 15U);
}

TEST_F(MemoryTest, DeviceViewTracksExternalMemory) {
  const BufferPtr device_buffer =
      Buffer::CreateOwnedBuffer(backend_, 8U, backend::MemoryKind::kDevice);
  ASSERT_NE(device_buffer, nullptr);
  const BufferPtr view =
      Buffer::CreateDeviceBufferView(backend_, device_buffer->device_address(), 8U);
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->memory_kind(), backend::MemoryKind::kDeviceView);
  EXPECT_EQ(view->device_address(), device_buffer->device_address());
}

TEST_F(MemoryTest, StreamWaitsForBackendEvent) {
  const BufferPtr buffer = Buffer::CreateOwnedBuffer(backend_, 16U, backend::MemoryKind::kDevice);
  ASSERT_NE(buffer, nullptr);
  const BackendStreamPtr producer = BackendStream::Create(backend_, backend::kDefaultStreamFlags);
  const BackendStreamPtr consumer = BackendStream::Create(backend_, backend::kDefaultStreamFlags);
  const BackendEventPtr event = BackendEvent::Create(backend_, backend::kDefaultEventFlags);
  ASSERT_NE(producer, nullptr);
  ASSERT_NE(consumer, nullptr);
  ASSERT_NE(event, nullptr);
  ASSERT_TRUE(buffer->FillAsync(19U, *producer));
  ASSERT_TRUE(event->Record(*producer));
  ASSERT_TRUE(consumer->WaitEvent(*event, backend::kDefaultEventWaitFlags));
  ASSERT_TRUE(buffer->FillAsync(20U, *consumer));
  ASSERT_TRUE(consumer->Synchronize());
  bool is_event_ready = false;
  ASSERT_TRUE(event->Query(&is_event_ready));
  EXPECT_TRUE(is_event_ready);
  std::vector<uint8_t> destination;
  ASSERT_TRUE(buffer->CopyDataToVector(&destination));
  EXPECT_EQ(destination, std::vector<uint8_t>(16U, 20U));
}

TEST_F(MemoryTest, DeviceBufferAsyncRoundTrip) {
  const BufferPtr buffer = Buffer::CreateOwnedBuffer(backend_, 16U, backend::MemoryKind::kDevice);
  ASSERT_NE(buffer, nullptr);
  const BackendStreamPtr stream = BackendStream::Create(backend_, backend::kDefaultStreamFlags);
  ASSERT_NE(stream, nullptr);
  const std::vector<uint64_t> source = {16ULL, 17ULL};
  std::vector<uint64_t> destination(source.size());
  ASSERT_TRUE(buffer->CopyFromHostAsync(source.data(), source.size() * sizeof(uint64_t), *stream));
  ASSERT_TRUE(stream->Synchronize());
  ASSERT_TRUE(buffer->CopyToHostAsync(buffer->size_bytes(), destination.data(), *stream));
  ASSERT_TRUE(stream->Synchronize());
  EXPECT_EQ(destination, source);
}

TEST_F(MemoryTest, DeviceBufferFill) {
  const BufferPtr buffer = Buffer::CreateOwnedBuffer(backend_, 16U, backend::MemoryKind::kDevice);
  ASSERT_NE(buffer, nullptr);
  ASSERT_TRUE(buffer->Fill(18U));
  std::vector<uint8_t> destination;
  ASSERT_TRUE(buffer->CopyDataToVector(&destination));
  EXPECT_EQ(destination, std::vector<uint8_t>(16U, 18U));
}

TEST_F(MemoryTest, CopyBetweenDeviceBuffers) {
  const BufferPtr source = Buffer::CreateOwnedBuffer(backend_, 8U, backend::MemoryKind::kDevice);
  const BufferPtr destination =
      Buffer::CreateOwnedBuffer(backend_, 8U, backend::MemoryKind::kDevice);
  ASSERT_NE(source, nullptr);
  ASSERT_NE(destination, nullptr);
  const std::vector<uint32_t> values = {19U, 20U};
  ASSERT_TRUE(source->CopyDataFromVector(values));
  ASSERT_TRUE(destination->CopyFromBuffer(*source));
  std::vector<uint32_t> result;
  ASSERT_TRUE(destination->CopyDataToVector(&result));
  EXPECT_EQ(result, values);
}

TEST_F(MemoryTest, ZeroSizeBufferIsRejected) {
  EXPECT_EQ(Buffer::CreateOwnedBuffer(nullptr, 0U, backend::MemoryKind::kHost), nullptr);
}

TEST_F(MemoryTest, TensorReshapePreservesElementCount) {
  const auto tensor =
      Tensor::Create(backend_, {2U, 3U}, DataType::kFloat32, backend::MemoryKind::kHost);
  ASSERT_NE(tensor, nullptr);
  EXPECT_TRUE(tensor->Reshape({3U, 2U}));
  EXPECT_FALSE(tensor->Reshape({4U, 2U}));
}

class OpenCLMemoryTest : public ::testing::Test {
protected:
  void SetUp() override {
    backend_ = backend::CreateBackend(backend::BackendKind::kOpenCL);
    if (backend_ == nullptr || !backend_->IsAvailable()) {
      GTEST_SKIP() << "OpenCL backend is unavailable";
    }
  }

  std::shared_ptr<backend::Backend> backend_;
};

TEST_F(OpenCLMemoryTest, DeviceBufferRoundTrip) {
  const BufferPtr buffer = Buffer::CreateOwnedBuffer(backend_, 16U, backend::MemoryKind::kDevice);
  const BackendStreamPtr stream = BackendStream::Create(backend_, backend::kDefaultStreamFlags);
  ASSERT_NE(buffer, nullptr);
  ASSERT_NE(stream, nullptr);
  const std::vector<uint32_t> source = {21U, 22U, 23U, 24U};
  std::vector<uint32_t> destination(source.size());
  ASSERT_TRUE(buffer->CopyFromHostAsync(source.data(), source.size() * sizeof(uint32_t), *stream));
  ASSERT_TRUE(stream->Synchronize());
  ASSERT_TRUE(buffer->CopyToHostAsync(buffer->size_bytes(), destination.data(), *stream));
  ASSERT_TRUE(stream->Synchronize());
  EXPECT_EQ(destination, source);
}

} // namespace
} // namespace memory
} // namespace omni_runtime
