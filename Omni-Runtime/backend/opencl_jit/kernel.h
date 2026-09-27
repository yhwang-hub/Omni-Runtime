#pragma once

#define CL_TARGET_OPENCL_VERSION 300

#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

#include <CL/cl.h>

#include "Omni-Runtime/backend/opencl_jit/options.h"
#include "Omni-Runtime/backend/opencl_jit/executor.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace opencl {

class Program final {
public:
  ~Program();
  cl_program native_program() const;

private:
  friend class Kernel;
  cl_program program_ = nullptr;
};

class Kernel final {
public:
  static bool BuildProgram(const cl_context context, const cl_device_id device,
                           const std::string &source, const CompilerOptions &options,
                           std::shared_ptr<Program> *const program);
  static bool BindArguments(const cl_kernel kernel,
                            const std::vector<::omni_runtime::backend::OpenCLArgument> &arguments);
  static bool Load(cl_context context, cl_device_id device,
                   const std::filesystem::path &source_path, const std::string &entry_name,
                   const CompilerOptions &options, std::shared_ptr<Kernel> *const kernel);
  ~Kernel();
  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;

  bool Launch(cl_command_queue queue, const std::vector<std::size_t> &global_size,
              const std::vector<std::size_t> &local_size,
              const std::vector<const void *> &arguments) const;
  bool Launch(const cl_command_queue queue, const std::vector<std::size_t> &global_size,
              const std::vector<std::size_t> &local_size,
              const std::vector<::omni_runtime::backend::OpenCLArgument> &arguments) const;

  cl_kernel native_kernel() const;

private:
  Kernel(cl_context context, cl_program program, cl_kernel kernel);
  cl_context context_ = nullptr;
  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;
  mutable std::mutex mutex_;
};

} // namespace opencl
} // namespace backend
} // namespace jit
} // namespace omni_runtime
