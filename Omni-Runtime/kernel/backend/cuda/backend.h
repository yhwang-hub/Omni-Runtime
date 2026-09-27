#pragma once

#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "Omni-Runtime/kernel/backend/cuda/kernel.h"
#include "Omni-Runtime/kernel/backend/cuda/options.h"
#include "Omni-Runtime/kernel/runtime/config.h"
#include "Omni-Runtime/kernel/runtime/runtime.h"
#include "Omni-Runtime/kernel/utils/command.h"
#include "Omni-Runtime/kernel/utils/env.h"
#include "Omni-Runtime/kernel/utils/exception.h"
#include "Omni-Runtime/kernel/utils/filesystem.h"
#include "Omni-Runtime/kernel/utils/gil.h"
#include "Omni-Runtime/kernel/utils/hash.h"
#include "Omni-Runtime/kernel/utils/json.h"
#include "Omni-Runtime/kernel/utils/str.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace cuda {

class CUDA {
public:
  using Device = cuda::Device;
  using Kernel = cuda::Kernel;
  using CompilerOptions = cuda::CompilerOptions;
  using LaunchOptions = cuda::LaunchOptions;

  struct CompilerInfo {
    std::filesystem::path path;
    std::string version;

    [[nodiscard]] std::string get_hash() const {
      return ::omni_runtime::jit::utils::hash::get_hex_digest(version);
    }

    [[nodiscard]] ::omni_runtime::jit::utils::json to_json() const {
      return ::omni_runtime::jit::utils::json::object_t{
          {"path", path.string()},
          {"version", version},
      };
    }
  };

  struct Toolkit {
    std::filesystem::path nvcc;
    std::optional<std::filesystem::path> cuobjdump;
  };

  Toolkit toolkit;
  CompilerInfo compiler_info;

  explicit CUDA(const ::omni_runtime::jit::utils::Env &env)
      : toolkit(find_cuda_toolkit(env)), compiler_info(get_compiler_info()) {
  }

  CompilerInfo get_compiler_info() const;

  void compile(const std::string &source, std::filesystem::path dir,
               const ::omni_runtime::jit::utils::Env &env,
               const ::omni_runtime::jit::backend::runtime::Config &config,
               const CompilerOptions &options) const;

  static std::shared_ptr<Kernel> load(const std::filesystem::path &dir,
                                      const ::omni_runtime::jit::utils::Env &env);

  static Toolkit find_cuda_toolkit(const ::omni_runtime::jit::utils::Env &env);
};

} // namespace cuda
} // namespace backend
} // namespace jit
} // namespace omni_runtime
