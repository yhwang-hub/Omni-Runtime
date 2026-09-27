#include <cstdint>
#include <array>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>

#include "gtest/gtest.h"

#include "Omni-Runtime/backend/backend.h"

namespace omni_runtime {
namespace backend {
namespace {

DeviceView View(const Allocation &allocation, const BackendKind kind) {
  return {.data = allocation.data,
          .offset_bytes = allocation.offset_bytes,
          .size_bytes = allocation.size_bytes,
          .backend_kind = kind};
}

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

TEST(BackendSelectionTest, HonorsExplicitSelectionAndRejectsUnknownNames) {
  for (const std::string name : {"cuda", "opencl", "auto"}) {
    const auto device = CreateComputeBackend(name);
    ASSERT_NE(device, nullptr) << name;
    EXPECT_TRUE(device->IsAvailable());
    if (name != "auto") {
      EXPECT_EQ(name, BackendName(device->kind()));
    }
  }
  ::testing::internal::CaptureStderr();
  EXPECT_EQ(CreateComputeBackend("unsupported_backend"), nullptr);
  EXPECT_NE(::testing::internal::GetCapturedStderr().find("unsupported_backend"),
            std::string::npos);
}

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

void CheckPreparedSequence(const std::shared_ptr<Backend> &backend_) {
  ASSERT_TRUE(backend_->InitializeCompute());
  Allocation state;
  Allocation weight;
  Allocation hidden;
  Allocation logits;
  ASSERT_TRUE(backend_->Allocate(MemoryKind::kDevice, 5U * sizeof(int32_t), &state));
  ASSERT_TRUE(backend_->Allocate(MemoryKind::kDevice, 8U * sizeof(float), &weight));
  ASSERT_TRUE(backend_->Allocate(MemoryKind::kDevice, 2U * sizeof(float), &hidden));
  ASSERT_TRUE(backend_->Allocate(MemoryKind::kDevice, 2U * sizeof(float), &logits));
  const std::array<int32_t, 5U> initial_state = {0, 4, 4, 4, 126};
  const std::array<float, 8U> embedding = {0, 1, 1, 0, 1, 0, 0, 1};
  ASSERT_TRUE(
      backend_->CopyHostToDevice(initial_state.data(), sizeof(initial_state), state, nullptr));
  ASSERT_TRUE(backend_->CopyHostToDevice(embedding.data(), sizeof(embedding), weight, nullptr));
  const DeviceView state_view = View(state, backend_->kind());
  const DeviceView weight_view = View(weight, backend_->kind());
  const DeviceView hidden_view = View(hidden, backend_->kind());
  const DeviceView logits_view = View(logits, backend_->kind());
  const std::vector<ComputeOperation> operations = {
      Embedding{.indices = state_view,
                .weight = weight_view.Slice(4U * sizeof(float), 4U * sizeof(float)),
                .token_count = 1,
                .hidden_size = 2,
                .output = hidden_view},
      Linear{.input = hidden_view,
             .weight = weight_view,
             .rows = 1,
             .inner = 2,
             .columns = 2,
             .output = logits_view},
      Argmax{.logits = logits_view, .rows = 1, .columns = 2, .output = state_view},
      AdvanceSequence{.positions = state_view.Slice(sizeof(int32_t), 3U * sizeof(int32_t)),
                      .cache_position = state_view.Slice(4U * sizeof(int32_t), sizeof(int32_t))}};
  std::unique_ptr<ExecutionPlan> plan;
  ASSERT_TRUE(backend_->CreateExecutionPlan(ScalarType::kFloat32, operations, state, &plan));
  std::array<int32_t, 5U> actual = {};
  ASSERT_TRUE(backend_->CopyDeviceToHost(state, sizeof(actual), actual.data(), nullptr));
  EXPECT_EQ(actual, initial_state);
  for (int32_t iteration = 1; iteration <= 4; ++iteration) {
    double elapsed_ms = 0.0;
    ASSERT_TRUE(plan->Run(sizeof(actual), actual.data(), &elapsed_ms));
    EXPECT_GT(elapsed_ms, 0.0);
    EXPECT_EQ(actual[0], iteration % 2);
    EXPECT_EQ(actual[1], 4 + iteration);
    EXPECT_EQ(actual[2], 4 + iteration);
    EXPECT_EQ(actual[3], 4 + iteration);
    EXPECT_EQ(actual[4], 126 + iteration);
  }
  ASSERT_TRUE(
      backend_->CopyHostToDevice(initial_state.data(), sizeof(initial_state), state, nullptr));
  const std::shared_ptr<Backend> second_backend = CreateBackend(backend_->kind());
  ASSERT_NE(second_backend, nullptr);
  ASSERT_TRUE(second_backend->InitializeCompute());
  double elapsed_ms = 0.0;
  ASSERT_TRUE(plan->Run(sizeof(actual), actual.data(), &elapsed_ms));
  EXPECT_EQ(actual[0], 1);
  EXPECT_EQ(actual[4], 127);
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(plan->Run(sizeof(int32_t), actual.data(), &elapsed_ms));
  EXPECT_FALSE(plan->Run(sizeof(actual), nullptr, &elapsed_ms));
  EXPECT_FALSE(plan->Run(sizeof(actual), actual.data(), nullptr));
  const std::string invalid_output_log = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(invalid_output_log.find("Invalid execution plan outputs or state"), std::string::npos);
  ASSERT_TRUE(plan->Run(sizeof(actual), actual.data(), &elapsed_ms));
  EXPECT_EQ(actual[0], 0);
  EXPECT_EQ(actual[4], 128);
  plan.reset();
  ASSERT_TRUE(backend_->Free(&state));
  ASSERT_TRUE(backend_->Free(&weight));
  ASSERT_TRUE(backend_->Free(&hidden));
  ASSERT_TRUE(backend_->Free(&logits));
}

TEST_F(CudaBackendTest, PreparesWithoutExecutingAndReplaysDeviceResidentSequence) {
  CheckPreparedSequence(backend_);
}

TEST_F(CudaBackendTest, PreparedSequenceWaitsForDefaultStreamCompute) {
  ASSERT_TRUE(backend_->InitializeCompute());
  std::array<Allocation, 4U> allocations;
  for (size_t index = 0U; index < allocations.size(); ++index) {
    ASSERT_TRUE(backend_->Allocate(MemoryKind::kDevice,
                                   index == 3U ? sizeof(int32_t) : 2U * sizeof(float),
                                   &allocations[index]));
  }
  const std::array<float, 2U> input = {0.0F, 3.0F};
  const std::array<float, 2U> zero = {};
  const std::array<float, 2U> initial = {2.0F, 1.0F};
  ASSERT_TRUE(backend_->CopyHostToDevice(input.data(), sizeof(input), allocations[0], nullptr));
  ASSERT_TRUE(backend_->CopyHostToDevice(zero.data(), sizeof(zero), allocations[1], nullptr));
  const Add producer{View(allocations[0], BackendKind::kCuda),
                     View(allocations[1], BackendKind::kCuda), 2U,
                     View(allocations[2], BackendKind::kCuda)};
  ASSERT_TRUE(backend_->Compute(ScalarType::kFloat32, producer));
  const std::vector<ComputeOperation> operations = {Argmax{
      View(allocations[2], BackendKind::kCuda), 1, 2, View(allocations[3], BackendKind::kCuda)}};
  std::unique_ptr<ExecutionPlan> plan;
  ASSERT_TRUE(
      backend_->CreateExecutionPlan(ScalarType::kFloat32, operations, allocations[3], &plan));
  ASSERT_TRUE(backend_->CopyHostToDevice(initial.data(), sizeof(initial), allocations[2], nullptr));
  ASSERT_EQ(cudaLaunchHostFunc(
                nullptr,
                [](void *) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); },
                nullptr),
            cudaSuccess);
  ASSERT_TRUE(backend_->Compute(ScalarType::kFloat32, producer));
  int32_t token = -1;
  double elapsed_ms = 0.0;
  ASSERT_TRUE(plan->Run(sizeof(token), &token, &elapsed_ms));
  EXPECT_EQ(token, 1);
  ASSERT_TRUE(backend_->SynchronizeStream(nullptr));
  plan.reset();
  for (auto &allocation : allocations) {
    ASSERT_TRUE(backend_->Free(&allocation));
  }
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
  ASSERT_TRUE(backend_->LinearMatmul(ScalarType::kFloat32, View(allocations[0], backend_->kind()),
                                     View(allocations[1], backend_->kind()), 2, 4, 3,
                                     View(allocations[2], backend_->kind())));
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(backend_->CopyDeviceToHost(allocations[2], actual.size() * sizeof(float),
                                         actual.data(), nullptr));
  EXPECT_EQ(actual, expected);
  ASSERT_TRUE(backend_->LinearMatmul(ScalarType::kFloat32, View(allocations[0], backend_->kind()),
                                     View(allocations[1], backend_->kind()), 1, 4, 3,
                                     View(allocations[2], backend_->kind())));
  ASSERT_TRUE(
      backend_->CopyDeviceToHost(allocations[2], 3U * sizeof(float), actual.data(), nullptr));
  EXPECT_EQ((std::vector<float>{actual[0], actual[1], actual[2]}), (std::vector<float>{5, 5, 10}));
  ASSERT_TRUE(backend_->Compute(ScalarType::kFloat32,
                                Add{.lhs = View(allocations[0], backend_->kind()),
                                    .rhs = View(allocations[1], backend_->kind()),
                                    .element_count = 4U,
                                    .output = View(allocations[2], backend_->kind())}));
  ASSERT_TRUE(
      backend_->CopyDeviceToHost(allocations[2], 4U * sizeof(float), actual.data(), nullptr));
  EXPECT_EQ((std::vector<float>{actual[0], actual[1], actual[2], actual[3]}),
            (std::vector<float>{2, 2, 3, 5}));
  EXPECT_FALSE(backend_->Compute(ScalarType::kFloat32,
                                 Add{.lhs = {},
                                     .rhs = View(allocations[1], backend_->kind()),
                                     .element_count = 4U,
                                     .output = View(allocations[2], backend_->kind())}));
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

TEST_F(OpenCLBackendTest, RejectsInvalidInferenceComputeExplicitly) {
  ASSERT_TRUE(backend_->InitializeCompute());
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(backend_->Compute(ScalarType::kFloat16,
                                 Add{.lhs = {}, .rhs = {}, .element_count = 1U, .output = {}}));
  EXPECT_NE(::testing::internal::GetCapturedStderr().find("Invalid add operation"),
            std::string::npos);
}

TEST_F(OpenCLBackendTest, PreparesWithoutExecutingAndReplaysDeviceResidentSequence) {
  CheckPreparedSequence(backend_);
}

} // namespace
} // namespace backend
} // namespace omni_runtime
