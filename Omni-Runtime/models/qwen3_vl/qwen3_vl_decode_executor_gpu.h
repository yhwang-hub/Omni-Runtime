#pragma once

#include <cstdint>
#include <string>

#include "Omni-Runtime/graph/element/node.h"
#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_define.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

struct Qwen3VLRuntime;
struct Qwen3VLGenerationContext;

class Qwen3VLDecodeExecutorGpu final : public graph::Node {
public:
  explicit Qwen3VLDecodeExecutorGpu(Qwen3VLRuntime *const runtime,
                                    Qwen3VLGenerationContext *const context);
  ~Qwen3VLDecodeExecutorGpu() final = default;
  Qwen3VLDecodeExecutorGpu(const Qwen3VLDecodeExecutorGpu &) = delete;
  Qwen3VLDecodeExecutorGpu &operator=(const Qwen3VLDecodeExecutorGpu &) = delete;

  bool Initialize();
  const std::string &error_message() const;

protected:
  graph::Status Run() final;

private:
  bool SetError(const std::string &message);

  Qwen3VLRuntime *runtime_ = nullptr;
  Qwen3VLGenerationContext *context_ = nullptr;
  std::string error_message_;
};

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
