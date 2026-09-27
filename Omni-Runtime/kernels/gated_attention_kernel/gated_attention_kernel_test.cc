#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kTokenCount = 8;
constexpr int32_t kHeadCount = 32;
constexpr int32_t kHeadSize = 128;
constexpr std::size_t kOutputCount = 32768U;
constexpr std::size_t kQueryGateCount = 65536U;

class GatedAttentionKernelTest : public test::KernelTestFixture {};

TEST_F(GatedAttentionKernelTest, SplitsQwen3VlGatedQuery) {
  std::vector<float> query_gate(kQueryGateCount);
  std::vector<float> expected_query(kOutputCount);
  std::vector<float> expected_gate(kOutputCount);
  for (std::size_t index = 0; index < kQueryGateCount; ++index) {
    query_gate[index] = 0.001f * static_cast<float>(index % kHeadSize) - 0.064f;
  }
  for (std::size_t index = 0; index < kOutputCount; ++index) {
    const std::size_t head_row = index / kHeadSize;
    const std::size_t dimension = index % kHeadSize;
    const std::size_t source = head_row * 2U * kHeadSize + dimension;
    expected_query[index] = query_gate[source];
    expected_gate[index] = query_gate[source + kHeadSize];
  }
  const auto query_gate_device =
      memory::Buffer::CreateDeviceBufferFromVector(backend(), query_gate);
  const auto query_device =
      memory::Buffer::CreateDeviceBuffer<float>(backend(), expected_query.size());
  const auto gate_device =
      memory::Buffer::CreateDeviceBuffer<float>(backend(), expected_gate.size());
  ASSERT_TRUE(Compile("split_gated_query_float", test::MakeSplitGatedQueryFloatSource()));
  ASSERT_TRUE(Launch(dim3(32, 1, 1), dim3(256, 1, 1), 0, query_gate_device->device_address(),
                     kTokenCount, kHeadCount, kHeadSize, query_device->device_address(),
                     gate_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual_query(expected_query.size());
  std::vector<float> actual_gate(expected_gate.size());
  ASSERT_TRUE(query_device->CopyDataToVector(&actual_query));
  ASSERT_TRUE(gate_device->CopyDataToVector(&actual_gate));
  EXPECT_TRUE(test::IsClose(actual_query, expected_query, 1.0e-6f) &&
              test::IsClose(actual_gate, expected_gate, 1.0e-6f));
}
} // namespace
} // namespace omni_runtime::kernels
