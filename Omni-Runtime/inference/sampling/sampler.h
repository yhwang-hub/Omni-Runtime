#pragma once

#include <cstddef>
#include <cstdint>

namespace omni_runtime {
namespace inference {

class GreedySampler final {
public:
  template <typename T>
  bool Sample(const T *const logits, const size_t vocabulary_size, int32_t *const token_id) const;
};

} // namespace inference
} // namespace omni_runtime

extern template bool omni_runtime::inference::GreedySampler::Sample<float>(
    const float *const logits, const size_t vocabulary_size, int32_t *const token_id) const;

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
