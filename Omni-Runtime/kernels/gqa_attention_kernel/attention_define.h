#pragma once

#include <cstdint>
#include <limits>

#include "kernels/kernel_define.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime::kernels::runtime {

struct AttentionArgs final {
  KernelLaunchConfig options;
  const void *const query;
  const void *const key_cache;
  const void *const value_cache;
  int32_t query_token_count;
  int32_t query_head_count;
  int32_t key_value_head_count;
  int32_t cached_token_count;
  int32_t start_position;
  int32_t head_dim;
  bool is_causal;
  void *const output;
  void *decode_partials = nullptr;
};

inline bool IsValidAttention(const AttentionArgs &args) {
  OMNI_RETURN_VAL_IF_LOG(
      args.query == nullptr || args.key_cache == nullptr || args.value_cache == nullptr ||
          args.output == nullptr || args.query_token_count <= 0 || args.cached_token_count <= 0 ||
          args.query_head_count <= 0 || args.key_value_head_count <= 0 ||
          args.query_head_count % args.key_value_head_count != 0 || args.head_dim <= 0 ||
          args.head_dim > 256 || args.start_position < 0 ||
          args.cached_token_count > std::numeric_limits<int32_t>::max() - 127 ||
          args.start_position > std::numeric_limits<int32_t>::max() - args.query_token_count - 127,
      false, ERROR, "Invalid attention tensors, dimensions or grouped heads");
  return true;
}

} // namespace omni_runtime::kernels::runtime
