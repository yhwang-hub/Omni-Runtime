#pragma once

#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"
#include "Omni-Runtime/inference/common/inference_define/inference_define.h"
#include "Omni-Runtime/memory/tensor.h"

namespace omni_runtime {
namespace inference {

class WeightLoader final {
public:
  bool LoadTensor(const util::GGUFSerializer &serializer, const std::string &name,
                  const ModelWeightType compute_type,
                  const std::shared_ptr<backend::Backend> &backend,
                  memory::TensorPtr *const tensor);
  bool LoadEncoded(const util::GGUFSerializer &serializer, const std::string &name,
                   const std::shared_ptr<backend::Backend> &backend,
                   memory::BufferPtr *const buffer);

  const std::string &error_message() const {
    return error_message_;
  }

private:
  bool SetError(const std::string &message);
  std::string error_message_;
};

} // namespace inference
} // namespace omni_runtime
