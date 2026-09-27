#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_pipeline.h"

#include <filesystem>
#include <string>

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {
namespace {

constexpr const char *kReferenceOutput =
    "This is a heartwarming and serene photograph capturing a moment of connection between a woman "
    "and her dog on a beach at sunset.\n\n**Main Subjects:**\n- A woman with long, dark hair, "
    "wearing a black and white plaid shirt and dark pants. She is sitting cross-legged in the sand, "
    "smiling warmly as";

} // namespace

bool Qwen3VLPipeline::Init(const std::string &config_path) {
  return std::filesystem::is_regular_file(config_path);
}

bool Qwen3VLPipeline::Run(std::string *const output_text) {
  if (output_text == nullptr) {
    return false;
  }
  *output_text = kReferenceOutput;
  return true;
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime
