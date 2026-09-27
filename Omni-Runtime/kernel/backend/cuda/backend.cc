#include "Omni-Runtime/kernel/backend/cuda/backend.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace cuda {

CUDA::CompilerInfo CUDA::get_compiler_info() const {
  const auto version = ::omni_runtime::jit::utils::call_external_command(
      toolkit.nvcc.string() + " --version", false);

  // Should support arch-family
  std::smatch match;
  JIT_HOST_ASSERT(std::regex_search(version, match, std::regex(R"(release (\d+)\.(\d+))")),
                  "failed to parse NVCC version from:\n{}", version);
  const int major = std::stoi(match[1].str());
  const int minor = std::stoi(match[2].str());
  JIT_HOST_ASSERT(major > 12 or (major == 12 and minor >= 9), "NVCC version must be at least 12.9");

  CompilerInfo compiler_info;
  compiler_info.path = toolkit.nvcc;
  compiler_info.version = version;
  return compiler_info;
}

void CUDA::compile(const std::string &source, std::filesystem::path dir,
                   const ::omni_runtime::jit::utils::Env &env,
                   const ::omni_runtime::jit::backend::runtime::Config &config,
                   const CompilerOptions &options) const {
  // Release GIL to let other Python threads run

  // Paths
  dir = std::filesystem::absolute(dir).lexically_normal();
  const auto source_path = dir / "kernel.cu";
  const auto cubin_path = dir / "kernel.cubin";
  const bool debug = env.get<bool>("JIT_DEBUG", false);
  const bool print_compiler_command = debug or env.get<bool>("JIT_PRINT_COMPILER_COMMAND", false);

  // Write source code
  ::omni_runtime::jit::utils::write_file_sync(source_path, source);

  // Build the full command
  std::vector<std::string> args = {
      toolkit.nvcc.string(), source_path.string(), "--cubin", "--output-file", cubin_path.string(),
  };
  const auto option_flags = options.get_flags();
  args.insert(args.end(), option_flags.begin(), option_flags.end());
  for (const auto &include_dir : config.include_dirs) {
    args.emplace_back("--include-path");
    args.emplace_back(include_dir.string());
  }
  const auto command = ::omni_runtime::jit::utils::str::join(args, " ");

  // NOTES: change directory into a temporary empty directory to prevent same name include files
  const auto cd_command = "cd " + dir.string() + " && ";

  // Compile
  const auto compiler_output = ::omni_runtime::jit::utils::call_external_command(
      cd_command + command, print_compiler_command);
  if (options.ptxas_verbose.value_or(false)) {
    std::fputs(compiler_output.c_str(), stdout);
    std::fflush(stdout);
  }
  JIT_HOST_ASSERT(
      not options.check_no_spills.value_or(false) or
          not std::regex_search(compiler_output,
                                std::regex(R"(spilled\s+to\s+local\s+memory)", std::regex::icase)),
      "PTXAS reported register spills:\n{}", compiler_output);
  JIT_HOST_ASSERT(not options.check_no_local_memory.value_or(false) or
                      not std::regex_search(compiler_output, std::regex(R"(local\s+memory\s+used)",
                                                                        std::regex::icase)),
                  "PTXAS reported local memory usage:\n{}", compiler_output);
  JIT_HOST_ASSERT(std::filesystem::is_regular_file(cubin_path) and
                      std::filesystem::file_size(cubin_path) != 0,
                  "NVCC did not produce a valid CUBIN: {}", cubin_path.string());

  // Run post hook
  if (options.post_hook) {
    const auto hook_path = config.get_python_path(*options.post_hook);
    const auto hook_command =
        "cd " + dir.string() + " && python " + hook_path.string() + " " + cubin_path.string();
    ::omni_runtime::jit::utils::call_external_command(hook_command, print_compiler_command);
  }

  // Dump PTX
  if (options.dump_ptx.value_or(false)) {
    const auto ptx_path = dir / "kernel.ptx";
    auto ptx_args = args;
    ptx_args[2] = "--ptx", ptx_args[4] = ptx_path.string();
    ::omni_runtime::jit::utils::call_external_command(
        cd_command + ::omni_runtime::jit::utils::str::join(ptx_args, " "), print_compiler_command);
    JIT_HOST_ASSERT(std::filesystem::is_regular_file(ptx_path) and
                        std::filesystem::file_size(ptx_path) != 0,
                    "NVCC did not produce a valid PTX: {}", ptx_path.string());
  }

  // Dump SASS
  if (options.dump_sass.value_or(false)) {
    JIT_HOST_ASSERT(toolkit.cuobjdump.has_value());
    const auto sass_path = dir / "kernel.sass";
    const auto sass_command = toolkit.cuobjdump->string() + " --dump-sass " + cubin_path.string();
    const auto sass = ::omni_runtime::jit::utils::call_external_command(cd_command + sass_command,
                                                                        print_compiler_command);
    JIT_HOST_ASSERT(not sass.empty(), "cuobjdump did not produce valid SASS for {}",
                    cubin_path.string());
    ::omni_runtime::jit::utils::write_file_sync(sass_path, sass);
  }

  // Write metadata
  const ::omni_runtime::jit::utils::json metadata = ::omni_runtime::jit::utils::json::object_t{
      {"command", command},
      {"config", config.to_json()},
      {"compiler_info", compiler_info.to_json()},
      {"compiler_options", options.to_json()},
  };
  ::omni_runtime::jit::utils::write_file_sync(dir / "meta.json", metadata.dump());
}

std::shared_ptr<Kernel> CUDA::load(const std::filesystem::path &dir,
                                   const ::omni_runtime::jit::utils::Env &env) {
  return Kernel::load(dir, env);
}

CUDA::Toolkit CUDA::find_cuda_toolkit(const ::omni_runtime::jit::utils::Env &env) {
  std::filesystem::path home_path;

  // Find CUDA home
  // 1. `CUDA_HOME` or `CUDA_PATH`
  if (home_path.empty()) {
    home_path = ::omni_runtime::jit::utils::get_env<std::string>("CUDA_HOME", "");
    home_path = home_path.empty()
                    ? std::filesystem::path(
                          ::omni_runtime::jit::utils::get_env<std::string>("CUDA_PATH", ""))
                    : home_path;
  }

  // 2. `which nvcc`
  if (home_path.empty()) {
    try {
      auto path = ::omni_runtime::jit::utils::call_external_command("which nvcc", false);
      while (not path.empty() and (path.back() == '\r' or path.back() == '\n')) {
        path.pop_back();
      }
      if (not path.empty()) {
        home_path = std::filesystem::path(path).parent_path().parent_path();
      }
    } catch (...) {
    }
  }

  // 3. /usr/local/cuda
  if (home_path.empty() and std::filesystem::exists("/usr/local/cuda")) {
    home_path = "/usr/local/cuda";
  }

  // Canonicalize
  JIT_HOST_ASSERT(not home_path.empty() and std::filesystem::exists(home_path));
  home_path = std::filesystem::absolute(home_path).lexically_normal();

  // Find NVCC
  std::filesystem::path nvcc;
  if (const auto path = env.get<std::string>("JIT_NVCC_COMPILER"); path and not path->empty()) {
    nvcc = std::filesystem::absolute(*path).lexically_normal();
  } else {
    nvcc = home_path / "bin/nvcc";
  }
  JIT_HOST_ASSERT(::omni_runtime::jit::utils::is_executable(nvcc),
                  "NVCC compiler is not executable: {}", nvcc.string());

  // Try to find `cuobjdump`
  const auto cuobjdump_path = home_path / "bin/cuobjdump";
  std::optional<std::filesystem::path> cuobjdump =
      ::omni_runtime::jit::utils::is_executable(cuobjdump_path) ? std::optional(cuobjdump_path)
                                                                : std::nullopt;
  return {.nvcc = std::move(nvcc), .cuobjdump = std::move(cuobjdump)};
}

} // namespace cuda
} // namespace backend
} // namespace jit
} // namespace omni_runtime
