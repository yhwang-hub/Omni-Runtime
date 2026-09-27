#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kQueryTokenCount = 4;
constexpr int32_t kQueryHeadCount = 32;
constexpr int32_t kKeyValueHeadCount = 8;
constexpr int32_t kCachedTokenCount = 8;
constexpr int32_t kStartPosition = 4;
constexpr int32_t kHeadDim = 128;
constexpr std::size_t kQueryElementCount = 16384U;
constexpr std::size_t kKeyValueElementCount = 8192U;

class GqaAttentionKernelTest : public test::KernelTestFixture {};

TEST_F(GqaAttentionKernelTest, AttendsQwen3VlGroupedHeads) {
  std::vector<float> query(kQueryElementCount);
  std::vector<float> key(kKeyValueElementCount);
  std::vector<float> value(kKeyValueElementCount);
  for (std::size_t index = 0; index < kQueryElementCount; ++index) {
    query[index] = 0.001f * static_cast<float>(index % kHeadDim) - 0.064f;
  }
  for (std::size_t index = 0; index < kKeyValueElementCount; ++index) {
    key[index] = 0.0005f * static_cast<float>(index % kHeadDim) - 0.032f;
    value[index] = 0.0002f * static_cast<float>(index % kHeadDim) + 0.016f;
  }
  const int32_t group_size = kQueryHeadCount / kKeyValueHeadCount;
  std::vector<float> expected(kQueryElementCount);
  for (int32_t query_row = 0; query_row < kQueryTokenCount * kQueryHeadCount; ++query_row) {
    const int32_t query_token = query_row / kQueryHeadCount;
    const int32_t query_head = query_row % kQueryHeadCount;
    const int32_t key_value_head = query_head / group_size;
    const int32_t source_limit = kStartPosition + query_token + 1;
    std::vector<float> scores(source_limit);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    for (int32_t source_token = 0; source_token < source_limit; ++source_token) {
      const int64_t key_offset =
          (static_cast<int64_t>(source_token) * kKeyValueHeadCount + key_value_head) * kHeadDim;
      float score = 0.0f;
      for (int32_t dimension = 0; dimension < kHeadDim; ++dimension) {
        score += query[static_cast<int64_t>(query_row) * kHeadDim + dimension] *
                 key[key_offset + dimension];
      }
      scores[source_token] = score * scale;
    }
    const float maximum = *std::max_element(scores.begin(), scores.end());
    float denominator = 0.0f;
    for (float &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (int32_t source_token = 0; source_token < source_limit; ++source_token) {
      const int64_t value_offset =
          (static_cast<int64_t>(source_token) * kKeyValueHeadCount + key_value_head) * kHeadDim;
      for (int32_t dimension = 0; dimension < kHeadDim; ++dimension) {
        expected[static_cast<int64_t>(query_row) * kHeadDim + dimension] +=
            scores[source_token] / denominator * value[value_offset + dimension];
      }
    }
  }
  const auto query_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), query);
  const auto key_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), key);
  const auto value_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), value);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("gqa_attention_float", test::MakeGqaAttentionFloatSource()));
  ASSERT_TRUE(Launch(dim3(kQueryTokenCount * kQueryHeadCount, 1, 1), dim3(128, 1, 1), 0,
                     query_device->device_address(), key_device->device_address(),
                     value_device->device_address(), kQueryTokenCount, kQueryHeadCount,
                     kKeyValueHeadCount, kCachedTokenCount, kStartPosition, kHeadDim, true,
                     output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 3.0e-5f));
}
} // namespace
} // namespace omni_runtime::kernels
