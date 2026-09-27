#pragma once

#define CL_TARGET_OPENCL_VERSION 300

#include <filesystem>
#include <memory>
#include <string>

#include <CL/cl.h>

#include "Omni-Runtime/kernel/backend/opencl/kernel.h"
#include "Omni-Runtime/kernel/backend/opencl/options.h"

namespace omni_runtime {
namespace jit {
namespace backend {
namespace opencl {

class OpenCL final {
public:
  using Device = cl_device_id;
  using Kernel = opencl::Kernel;
  using CompilerOptions = opencl::CompilerOptions;

  OpenCL();
  ~OpenCL();
  OpenCL(const OpenCL &) = delete;
  OpenCL &operator=(const OpenCL &) = delete;

  bool IsAvailable() const;
  Device device() const;
  cl_context context() const;
  CompilerOptions default_compiler_options();

  bool Compile(const std::string &source, const std::filesystem::path &source_path,
               const std::string &entry_name, const CompilerOptions &options,
               std::shared_ptr<Kernel> *const kernel);

private:
  struct Runtime;
  std::unique_ptr<Runtime> runtime_;
};

} // namespace opencl
} // namespace backend
} // namespace jit
} // namespace omni_runtime
