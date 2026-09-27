#pragma once

#define CL_TARGET_OPENCL_VERSION 300

#include <optional>
#include <string>
#include <vector>

#include <CL/cl.h>

#include "Omni-Runtime/kernel/utils/env.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace opencl {

struct CompilerOptions {
  std::optional<std::string> optimize_level;
  std::optional<bool> is_debug_enabled;
  std::optional<std::vector<std::string>> extra_flags;

  static CompilerOptions default_options(const utils::Env &env) {
    return {
        .optimize_level = "fast",
        .is_debug_enabled = env.get<bool>("JIT_DEBUG", false),
        .extra_flags = std::vector<std::string>{},
    };
  }

  [[nodiscard]] CompilerOptions override_with(const CompilerOptions &overrides) const {
    CompilerOptions result = *this;
    if (overrides.optimize_level) {
      result.optimize_level = overrides.optimize_level;
    }
    if (overrides.is_debug_enabled) {
      result.is_debug_enabled = overrides.is_debug_enabled;
    }
    if (overrides.extra_flags) {
      result.extra_flags = overrides.extra_flags;
    }
    return result;
  }

  [[nodiscard]] std::vector<std::string> get_flags() const {
    std::vector<std::string> flags;
    if (optimize_level) {
      flags.emplace_back("-cl-fast-relaxed-math");
    }
    if (is_debug_enabled.value_or(false)) {
      flags.emplace_back("-g");
    }
    if (extra_flags) {
      flags.insert(flags.end(), extra_flags->begin(), extra_flags->end());
    }
    return flags;
  }
};

} // namespace opencl
} // namespace backend
} // namespace jit
} // namespace omni_runtime
