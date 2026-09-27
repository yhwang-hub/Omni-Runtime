#pragma once

#include <functional>
#include <string>

#include "Omni-Runtime/inference/tokenizer/tokenizer.h"
#include "Omni-Runtime/inference/common/inference_define/inference_define.h"

namespace omni_runtime {
namespace inference {

class AutoregressiveRunner final {
public:
  using DecodeStep = std::function<bool(int32_t *const, double *const)>;

  static bool Run(const GenerationRequest &request, const int32_t first_token,
                  const int32_t vocabulary_size, const int32_t end_of_sequence_id,
                  const int32_t end_of_text_id, const Tokenizer &tokenizer,
                  const DecodeStep &decode_step, GenerationResult *const result,
                  std::string *const error_message);
};

} // namespace inference
} // namespace omni_runtime
