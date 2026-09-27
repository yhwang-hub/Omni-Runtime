#include "kernels/kernel_test_utils.h"

#include <filesystem>
#include <memory>
#include <cmath>
#include <limits>
#include <string>

#include "Omni-Runtime/kernel/runtime/jit.h"
#include "Omni-Runtime/memory/buffer.h"

namespace omni_runtime {
namespace kernels {
namespace test {
namespace {

std::filesystem::path FindLibraryRoot() {
  std::error_code error;
  std::filesystem::path current_directory = std::filesystem::current_path(error);
  while (!error) {
    const std::filesystem::path candidate = current_directory / "Omni-Runtime" / "kernels";
    if (std::filesystem::is_directory(candidate, error)) {
      return current_directory / "Omni-Runtime";
    }
    error.clear();
    const std::filesystem::path parent_directory = current_directory.parent_path();
    if (parent_directory == current_directory) {
      break;
    }
    current_directory = parent_directory;
  }
  return std::filesystem::current_path() / "Omni-Runtime";
}

} // namespace

bool InitJit() {
  return ::omni_runtime::kernel::jit::JitRuntime::Init(FindLibraryRoot());
}

bool CompileKernel(const std::string &tag, const std::string &source,
                   std::shared_ptr<Kernel> *const kernel) {
  if (kernel == nullptr) {
    return false;
  }
  return ::omni_runtime::kernel::jit::JitRuntime::Compile(tag, source, kernel);
}

bool SyncDevice() {
  return cudaDeviceSynchronize() == cudaSuccess;
}

bool ReferenceAttention(const std::vector<float> &query, const std::vector<float> &key,
                        const std::vector<float> &value, const int32_t head_count,
                        const int32_t token_count, const int32_t head_dim, const bool is_causal,
                        std::vector<float> *const output) {
  if (output == nullptr || head_count <= 0 || token_count <= 0 || head_dim <= 0) {
    return false;
  }
  output->assign(static_cast<std::size_t>(head_count) * token_count * head_dim, 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (int32_t head = 0; head < head_count; ++head) {
    for (int32_t query_token = 0; query_token < token_count; ++query_token) {
      const int32_t source_limit = is_causal ? query_token + 1 : token_count;
      const int64_t query_offset =
          (static_cast<int64_t>(head) * token_count + query_token) * head_dim;
      float maximum = -std::numeric_limits<float>::infinity();
      std::vector<float> scores(source_limit);
      for (int32_t source_token = 0; source_token < source_limit; ++source_token) {
        const int64_t key_offset =
            (static_cast<int64_t>(head) * token_count + source_token) * head_dim;
        float score = 0.0f;
        for (int32_t dimension = 0; dimension < head_dim; ++dimension) {
          score += query[query_offset + dimension] * key[key_offset + dimension];
        }
        scores[source_token] = score * scale;
        maximum = std::max(maximum, scores[source_token]);
      }
      float denominator = 0.0f;
      for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
      }
      for (int32_t source_token = 0; source_token < source_limit; ++source_token) {
        const int64_t value_offset =
            (static_cast<int64_t>(head) * token_count + source_token) * head_dim;
        for (int32_t dimension = 0; dimension < head_dim; ++dimension) {
          (*output)[query_offset + dimension] +=
              scores[source_token] / denominator * value[value_offset + dimension];
        }
      }
    }
  }
  return true;
}

} // namespace test
} // namespace kernels
} // namespace omni_runtime
