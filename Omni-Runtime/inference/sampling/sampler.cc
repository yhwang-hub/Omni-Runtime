#include "Omni-Runtime/inference/sampling/sampler.h"

#include <climits>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {

template <typename T>
bool GreedySampler::Sample(const T *const logits, const size_t vocabulary_size,
                           int32_t *const token_id) const {
  OMNI_RETURN_VAL_IF(logits == nullptr || token_id == nullptr || vocabulary_size == 0U, false);
  size_t maximum_index = 0U;
  for (size_t index = 1U; index < vocabulary_size; ++index) {
    if (logits[index] > logits[maximum_index]) {
      maximum_index = index;
    }
  }
  OMNI_RETURN_VAL_IF(maximum_index > static_cast<size_t>(INT32_MAX), false);
  *token_id = static_cast<int32_t>(maximum_index);
  return true;
}

template bool GreedySampler::Sample<float>(const float *const logits, const size_t vocabulary_size,
                                           int32_t *const token_id) const;

} // namespace inference
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
