#include "Omni-Runtime/inference/common/text_stream/text_stream.h"

#include <exception>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime::inference {

TextStream::TextStream(const TextStreamCallback &callback) : callback_(callback) {
}

bool TextStream::Append(const std::string &bytes) {
  OMNI_RETURN_VAL_IF_LOG(is_cancelled_, false, ERROR, "Cannot append to a cancelled text stream");
  pending_ += bytes;
  return Emit(false);
}

bool TextStream::Finish() {
  return is_cancelled_ || Emit(true);
}

bool TextStream::Emit(const bool is_final) {
  std::string chunk;
  size_t offset = 0U;
  while (offset < pending_.size()) {
    const auto first = static_cast<unsigned char>(pending_[offset]);
    const size_t count = first < 0x80U                      ? 1U
                         : first >= 0xC2U && first <= 0xDFU ? 2U
                         : first >= 0xE0U && first <= 0xEFU ? 3U
                         : first >= 0xF0U && first <= 0xF4U ? 4U
                                                            : 0U;
    size_t valid = 1U;
    while (valid < count && offset + valid < pending_.size()) {
      const auto byte = static_cast<unsigned char>(pending_[offset + valid]);
      if (byte < 0x80U || byte > 0xBFU ||
          (valid == 1U && ((first == 0xE0U && byte < 0xA0U) || (first == 0xEDU && byte > 0x9FU) ||
                           (first == 0xF0U && byte < 0x90U) || (first == 0xF4U && byte > 0x8FU)))) {
        break;
      }
      ++valid;
    }
    if (count > valid && offset + valid == pending_.size() && !is_final) {
      break;
    }
    if (count == valid) {
      chunk.append(pending_, offset, count);
      offset += count;
    } else {
      chunk += "\xEF\xBF\xBD";
      offset += valid;
    }
  }
  pending_.erase(0U, offset);
  text_ += chunk;
  if (!chunk.empty() && callback_) {
    try {
      is_cancelled_ = !callback_(chunk);
    } catch (const std::exception &error) {
      error_message_ = std::string("Text stream callback failed: ") + error.what();
      LOG_ERROR(error_message_);
      return false;
    } catch (...) {
      error_message_ = "Text stream callback threw a non-standard exception";
      LOG_ERROR(error_message_);
      return false;
    }
  }
  return true;
}

bool TextStream::is_cancelled() const {
  return is_cancelled_;
}

const std::string &TextStream::text() const {
  return text_;
}

const std::string &TextStream::error_message() const {
  return error_message_;
}

} // namespace omni_runtime::inference
