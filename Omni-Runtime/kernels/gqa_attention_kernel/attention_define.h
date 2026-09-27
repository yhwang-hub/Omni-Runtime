#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

#include "kernels/kernel_define.h"
#include "Omni-Runtime/backend/cuda_jit/launch_define.h"
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
  const int32_t *cache_position = nullptr;
  bool is_fp16_key_value = false;
};

inline int32_t GgmlAttentionStreamGroups(const int32_t query_tokens, const int32_t cached_tokens,
                                         const int32_t kv_heads, const int32_t compute_units) {
  if (query_tokens <= 4 || cached_tokens <= 0 || kv_heads <= 0 || compute_units <= 0) {
    return 0;
  }
  const int64_t tiles = ((static_cast<int64_t>(query_tokens) + 7) / 8) * kv_heads;
  const int64_t key_tiles = ((static_cast<int64_t>(cached_tokens) + 255) / 256) * 8;
  // The reference D=256, 8x8 MMA tile is register-limited to two blocks per SM.
  const int64_t groups =
      std::min<int64_t>(static_cast<int64_t>(compute_units) * 2, tiles * key_tiles);
  const int64_t rounded = groups / tiles * tiles;
  return static_cast<int32_t>(rounded > 0 && 100 * (groups - rounded) / groups <= 5 ? rounded
                                                                                    : groups);
}

inline bool IsValidAttentionShape(const int32_t query_tokens, const int32_t query_heads,
                                  const int32_t kv_heads, const int32_t cached_tokens,
                                  const int32_t head_dim) {
  return query_tokens > 0 && query_heads > 0 && kv_heads > 0 && cached_tokens > 0 &&
         query_heads % kv_heads == 0 && head_dim > 0 && head_dim <= 256 &&
         uint64_t(query_tokens) * query_heads <= std::numeric_limits<uint64_t>::max() / head_dim &&
         uint64_t(cached_tokens) * kv_heads <= std::numeric_limits<uint64_t>::max() / head_dim;
}

inline bool IsValidAttention(const AttentionArgs &args) {
  OMNI_RETURN_VAL_IF_LOG(
      args.query == nullptr || args.key_cache == nullptr || args.value_cache == nullptr ||
          args.output == nullptr ||
          !IsValidAttentionShape(args.query_token_count, args.query_head_count,
                                 args.key_value_head_count, args.cached_token_count,
                                 args.head_dim) ||
          args.start_position < 0 ||
          args.cached_token_count > std::numeric_limits<int32_t>::max() - 127 ||
          args.start_position >
              std::numeric_limits<int32_t>::max() - args.query_token_count - 127 ||
          (args.cache_position != nullptr &&
           (args.query_token_count != 1 || args.decode_partials == nullptr || !args.is_causal)),
      false, ERROR, "Invalid attention tensors, dimensions or grouped heads");
  return true;
}

} // namespace omni_runtime::kernels::runtime
