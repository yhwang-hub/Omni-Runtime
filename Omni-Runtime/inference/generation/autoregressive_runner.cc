#include "Omni-Runtime/inference/generation/autoregressive_runner.h"

#include "Omni-Runtime/inference/common/text_stream/text_stream.h"
#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {

bool AutoregressiveRunner::Run(const GenerationRequest &request, const int32_t first_token,
                               const int32_t vocabulary_size, const int32_t end_of_sequence_id,
                               const int32_t end_of_text_id, const Tokenizer &tokenizer,
                               const DecodeStep &decode_step, GenerationResult *const result,
                               std::string *const error_message) {
  OMNI_RETURN_VAL_IF_LOG(result == nullptr || error_message == nullptr, false, ERROR,
                         "Generation outputs are null");
  OMNI_RETURN_VAL_IF_LOG(!decode_step || request.max_new_tokens <= 0 || vocabulary_size <= 0,
                         (*error_message = "invalid autoregressive inputs", false), ERROR,
                         "Invalid autoregressive inputs");
  error_message->clear();
  TextStream stream(request.stream_callback);
  int32_t token = first_token;
  for (int32_t index = 0; index < request.max_new_tokens; ++index) {
    OMNI_RETURN_VAL_IF(token < 0 || token >= vocabulary_size,
                       (*error_message = "sampled token ID is outside the vocabulary", false));
    result->generated_token_ids.push_back(token);
    if (token == end_of_sequence_id || token == end_of_text_id) {
      break;
    }
    std::string bytes;
    OMNI_RETURN_VAL_IF(!tokenizer.Decode({token}, &bytes),
                       (*error_message = tokenizer.error_message(), false));
    OMNI_RETURN_VAL_IF(!stream.Append(bytes), (*error_message = stream.error_message(), false));
    if (stream.is_cancelled() || index + 1 == request.max_new_tokens) {
      break;
    }
    double elapsed_ms = 0.0;
    OMNI_RETURN_VAL_IF(!decode_step(&token, &elapsed_ms),
                       (*error_message = "decode step failed", false));
    result->decode_time_ms += elapsed_ms;
    ++result->decode_token_count;
  }
  OMNI_RETURN_VAL_IF(!stream.Finish(), (*error_message = stream.error_message(), false));
  result->text = stream.text();
  result->is_cancelled = stream.is_cancelled();
  return true;
}

} // namespace inference
} // namespace omni_runtime
