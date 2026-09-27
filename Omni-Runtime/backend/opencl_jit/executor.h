#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "Omni-Runtime/backend/backend_define.h"

namespace omni_runtime::backend {

struct OpenCLArgument final {
  enum class Kind : uint8_t { kBuffer, kScalar, kLocal };
  Kind kind = Kind::kScalar;
  DeviceView buffer;
  std::array<std::byte, 16U> value = {};
  size_t size_bytes = 0U;
};

struct OpenCLInvocation final {
  std::string source;
  std::string entry;
  ScalarType type = ScalarType::kFloat32;
  std::array<size_t, 3U> global_size = {1U, 1U, 1U};
  std::array<size_t, 3U> local_size = {1U, 1U, 1U};
  uint32_t dimensions = 1U;
  std::array<int32_t, 3U> tensor_shape = {};
  bool has_nvidia_mma = false;
  int32_t compute_unit_count = 0;
  std::vector<std::string> build_defines;
  std::vector<OpenCLArgument> arguments;

  void Buffer(const DeviceView &view) {
    arguments.push_back({.kind = OpenCLArgument::Kind::kBuffer, .buffer = view});
    Scalar(static_cast<uint64_t>(view.offset_bytes));
  }

  template <typename T> void Scalar(const T value) {
    static_assert(std::is_arithmetic_v<T> && sizeof(T) <= 16U);
    OpenCLArgument argument;
    argument.size_bytes = sizeof(T);
    std::memcpy(argument.value.data(), &value, sizeof(T));
    arguments.push_back(argument);
  }

  void Local(const size_t size_bytes) {
    arguments.push_back({.kind = OpenCLArgument::Kind::kLocal, .size_bytes = size_bytes});
  }
};

class OpenCLExecutor final {
public:
  class PreparedKernel;
  using PreparedSequence = std::vector<std::shared_ptr<PreparedKernel>>;

  OpenCLExecutor();
  ~OpenCLExecutor();
  OpenCLExecutor(const OpenCLExecutor &) = delete;
  OpenCLExecutor &operator=(const OpenCLExecutor &) = delete;

  bool Initialize(void *const context, void *const device, void *const queue);
  bool Launch(const OpenCLInvocation &invocation);
  bool BeginPreparation(PreparedSequence *const sequence);
  bool EndPreparation();
  bool Enqueue(const PreparedSequence &sequence);
  bool HasNvidiaMma() const;
  int32_t compute_unit_count() const;

private:
  struct Runtime;
  std::unique_ptr<Runtime> runtime_;
};

} // namespace omni_runtime::backend
