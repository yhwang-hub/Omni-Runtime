#pragma once

#include <algorithm>
#include <array>
#include <type_traits>

#include "kernels/kernel_test_utils.h"
#include "kernels/gqa_attention_kernel/attention_define.h"

namespace omni_runtime::kernels::test {

template <typename T, template <typename> class Implementation>
void CheckFlashAttention(const std::shared_ptr<backend::Backend> &backend) {
  const std::array<std::array<int32_t, 4U>, 8U> shapes = {{{17, 35, 72, 0},
                                                           {67, 137, 128, 70},
                                                           {3, 133, 77, 1},
                                                           {1, 257, 128, 256},
                                                           {33, 129, 80, 0},
                                                           {3, 5, 1, 0},
                                                           {17, 65, 96, 0},
                                                           {5, 19, 256, 0}}};
  for (const auto &shape : shapes) {
    const int32_t tokens = shape[0];
    const int32_t cached = shape[1];
    const int32_t dimensions = shape[2];
    if (dimensions > Implementation<T>::kMaxHeadDim) {
      continue;
    }
    constexpr int32_t kHeads = 4;
    constexpr int32_t kKvHeads = 2;
    SCOPED_TRACE(::testing::Message()
                 << "tokens=" << tokens << ", cached=" << cached << ", dimensions=" << dimensions);
    std::vector<T> query(static_cast<size_t>(tokens) * kHeads * dimensions);
    std::vector<T> key(static_cast<size_t>(cached) * kKvHeads * dimensions);
    std::vector<T> value(key.size());
    const auto convert = [](const float number) {
      if constexpr (std::is_same_v<T, float>) {
        return number;
      } else {
        return LowPrecision<T>(number);
      }
    };
    const auto to_float = [](const T number) {
      if constexpr (std::is_same_v<T, float>) {
        return number;
      } else {
        return HostFloat(number);
      }
    };
    for (size_t index = 0U; index < query.size(); ++index) {
      query[index] = convert((static_cast<int32_t>((index * 7U) % 31U) - 15) / 16.0F);
    }
    for (size_t index = 0U; index < key.size(); ++index) {
      key[index] = convert((static_cast<int32_t>((index * 11U) % 29U) - 14) / 16.0F);
      value[index] = convert((static_cast<int32_t>((index * 13U) % 37U) - 18) / 16.0F);
      if (cached == 129) {
        key[index] = convert(-32.0F);
      }
    }
    std::vector<float> expected(query.size());
    for (int32_t token = 0; token < tokens; ++token) {
      for (int32_t head = 0; head < kHeads; ++head) {
        const int32_t limit = shape[3] > 0 ? std::min(cached, shape[3] + token + 1) : cached;
        const int64_t q_offset = (static_cast<int64_t>(token) * kHeads + head) * dimensions;
        std::vector<double> probabilities(limit);
        for (int32_t source = 0; source < limit; ++source) {
          const int64_t offset = (static_cast<int64_t>(source) * kKvHeads + head / 2) * dimensions;
          double score = 0.0;
          for (int32_t d = 0; d < dimensions; ++d) {
            score += static_cast<double>(to_float(query[q_offset + d])) * to_float(key[offset + d]);
          }
          probabilities[source] = score / std::sqrt(static_cast<double>(dimensions));
        }
        const double maximum = *std::max_element(probabilities.begin(), probabilities.end());
        double denominator = 0.0;
        for (double &probability : probabilities) {
          probability = std::exp(probability - maximum);
          denominator += probability;
        }
        for (int32_t d = 0; d < dimensions; ++d) {
          double weighted = 0.0;
          for (int32_t source = 0; source < limit; ++source) {
            const int64_t offset =
                (static_cast<int64_t>(source) * kKvHeads + head / 2) * dimensions + d;
            weighted += probabilities[source] * to_float(value[offset]);
          }
          expected[q_offset + d] = weighted / denominator;
        }
      }
    }
    const auto q = memory::Buffer::CreateDeviceBufferFromVector(backend, query);
    const auto k = memory::Buffer::CreateDeviceBufferFromVector(backend, key);
    const auto v = memory::Buffer::CreateDeviceBufferFromVector(backend, value);
    const auto output = memory::Buffer::CreateDeviceBuffer<T>(backend, query.size());
    ASSERT_NE(q, nullptr);
    ASSERT_NE(k, nullptr);
    ASSERT_NE(v, nullptr);
    ASSERT_NE(output, nullptr);
    const runtime::AttentionArgs args = {.options = {},
                                         .query = q->device_address(),
                                         .key_cache = k->device_address(),
                                         .value_cache = v->device_address(),
                                         .query_token_count = tokens,
                                         .query_head_count = kHeads,
                                         .key_value_head_count = kKvHeads,
                                         .cached_token_count = cached,
                                         .start_position = shape[3],
                                         .head_dim = dimensions,
                                         .is_causal = shape[3] > 0,
                                         .output = output->device_address()};
    ASSERT_TRUE(Implementation<T>::CompileAndLaunch("flash_attention_correctness", args));
    if (tokens == 17 && cached == 35) {
      runtime::AttentionArgs invalid = args;
      invalid.key_value_head_count = 3;
      EXPECT_FALSE(Implementation<T>::CompileAndLaunch("flash_invalid_group", invalid));
      invalid.key_value_head_count = kKvHeads;
      invalid.head_dim = 0;
      EXPECT_FALSE(Implementation<T>::CompileAndLaunch("flash_invalid_dimension", invalid));
      invalid.head_dim = dimensions;
      invalid.start_position = std::numeric_limits<int32_t>::max();
      EXPECT_FALSE(Implementation<T>::CompileAndLaunch("flash_invalid_position", invalid));
    }
    std::vector<T> actual(query.size());
    ASSERT_TRUE(output->CopyToHost(actual.size() * sizeof(T), actual.data()));
    constexpr float kTolerance = std::is_same_v<T, float>     ? 2.0e-5F
                                 : std::is_same_v<T, Float16> ? 2.0e-3F
                                                              : 1.5e-2F;
    for (size_t index = 0U; index < actual.size(); ++index) {
      ASSERT_TRUE(std::isfinite(to_float(actual[index])));
      ASSERT_NEAR(to_float(actual[index]), expected[index], kTolerance) << "index=" << index;
    }
  }
}

} // namespace omni_runtime::kernels::test
