#pragma once

#include <string>

#include "Omni-Runtime/inference/common/inference_define/inference_define.h"

namespace omni_runtime::inference {

class TextStream final {
public:
  explicit TextStream(const TextStreamCallback &callback);
  bool Append(const std::string &bytes);
  bool Finish();
  bool is_cancelled() const;
  const std::string &text() const;
  const std::string &error_message() const;

private:
  bool Emit(const bool is_final);
  TextStreamCallback callback_;
  std::string pending_;
  std::string text_;
  std::string error_message_;
  bool is_cancelled_ = false;
};

} // namespace omni_runtime::inference
