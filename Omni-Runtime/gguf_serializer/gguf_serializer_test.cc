#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

namespace omni_runtime {
namespace util {
namespace {

std::string ModelDirectory() {
  const char *const model_directory = std::getenv("QWEN3_VL_GGUF_MODEL_DIR");
  if (model_directory != nullptr && model_directory[0] != '\0') {
    return model_directory;
  }
  return "model/qwen3_vl";
}

std::string ModelPath(const std::string &file_name) {
  return (std::filesystem::path(ModelDirectory()) / file_name).string();
}

TEST(GGUFSerializerTest, ParsesQwen3VLModelsWithoutLoadingAllWeights) {
  struct ModelExpectation final {
    std::string file_name;
    std::string architecture;
    std::string model_type;
    uint64_t tensor_count = 0U;
    uint64_t metadata_count = 0U;
    uint64_t data_offset_bytes = 0U;
    std::unordered_map<uint32_t, size_t> tensor_type_counts;
  };

  const std::vector<ModelExpectation> expectations = {
      {"Qwen3VL-8B-Instruct-F16.gguf",
       "qwen3vl",
       "model",
       399U,
       30U,
       5957728U,
       {{static_cast<uint32_t>(GGUFTensorType::FLOAT32), 145U},
        {static_cast<uint32_t>(GGUFTensorType::FLOAT16), 254U}}},
      {"mmproj-Qwen3VL-8B-Instruct-F16.gguf",
       "clip",
       "mmproj",
       352U,
       23U,
       20608U,
       {{static_cast<uint32_t>(GGUFTensorType::FLOAT32), 234U},
        {static_cast<uint32_t>(GGUFTensorType::FLOAT16), 118U}}},
  };

  for (const ModelExpectation &expectation : expectations) {
    const std::string model_path = ModelPath(expectation.file_name);
    ASSERT_TRUE(std::filesystem::is_regular_file(model_path)) << model_path;
    ASSERT_TRUE(GGUFSerializer::IsGGUFFile(model_path)) << model_path;

    GGUFSerializer serializer;
    ASSERT_TRUE(serializer.Parse(model_path))
        << expectation.file_name << ": " << serializer.error_message();
    ASSERT_TRUE(serializer.IsParsed());
    const GGUFModelInfo &model = serializer.model();
    EXPECT_EQ(model.header.version, kGGUFVersion3);
    EXPECT_EQ(model.header.tensor_count, expectation.tensor_count);
    EXPECT_EQ(model.header.metadata_count, expectation.metadata_count);
    EXPECT_EQ(model.metadata.size(), expectation.metadata_count);
    EXPECT_EQ(model.tensors.size(), expectation.tensor_count);
    EXPECT_EQ(model.alignment_bytes, kGGUFDefaultAlignmentBytes);
    EXPECT_EQ(model.data_offset_bytes, expectation.data_offset_bytes);
    EXPECT_EQ(model.file_size_bytes, std::filesystem::file_size(model_path));

    std::string architecture;
    std::string model_type;
    ASSERT_TRUE(serializer.GetMetadata("general.architecture", &architecture));
    ASSERT_TRUE(serializer.GetMetadata("general.type", &model_type));
    EXPECT_EQ(architecture, expectation.architecture);
    EXPECT_EQ(model_type, expectation.model_type);

    std::unordered_map<uint32_t, size_t> tensor_type_counts;
    const GGUFTensorInfo *first_float32_tensor = nullptr;
    for (const GGUFTensorInfo &tensor : model.tensors) {
      ++tensor_type_counts[tensor.type_id];
      EXPECT_FALSE(tensor.name.empty());
      EXPECT_FALSE(tensor.dimensions.empty());
      EXPECT_GT(tensor.element_count, 0U);
      EXPECT_LE(tensor.data_offset_bytes, model.file_size_bytes);
      EXPECT_LE(tensor.storage_size_bytes, tensor.storage_span_bytes);
      EXPECT_LE(tensor.storage_span_bytes, model.file_size_bytes - tensor.data_offset_bytes);
      if (first_float32_tensor == nullptr &&
          tensor.type_id == static_cast<uint32_t>(GGUFTensorType::FLOAT32)) {
        first_float32_tensor = &tensor;
      }
    }
    EXPECT_EQ(tensor_type_counts, expectation.tensor_type_counts);

    ASSERT_FALSE(model.tensors.empty());
    const GGUFTensorInfo &first_tensor = model.tensors.front();
    const size_t preview_size_bytes =
        static_cast<size_t>(std::min<uint64_t>(64U, first_tensor.storage_size_bytes));
    std::vector<uint8_t> preview;
    ASSERT_TRUE(serializer.ReadTensorBytes(first_tensor.name, 0U, preview_size_bytes, &preview));
    EXPECT_EQ(preview.size(), preview_size_bytes);

    ASSERT_NE(first_float32_tensor, nullptr);
    const size_t value_count =
        static_cast<size_t>(std::min<uint64_t>(4U, first_float32_tensor->element_count));
    std::vector<float> float_values;
    ASSERT_TRUE(
        serializer.ReadTensorData(first_float32_tensor->name, 0U, value_count, &float_values));
    EXPECT_EQ(float_values.size(), value_count);
  }
}

TEST(GGUFSerializerTest, SupportsScalarLowPrecisionAndExtensibleTensorTypes) {
  GGUFTensorTypeInfo type_info;
  ASSERT_TRUE(GGUFSerializer::GetTensorTypeInfo(static_cast<uint32_t>(GGUFTensorType::FLOAT32),
                                                &type_info));
  EXPECT_EQ(type_info.block_element_count, 1U);
  EXPECT_EQ(type_info.block_size_bytes, sizeof(float));

  ASSERT_TRUE(GGUFSerializer::GetTensorTypeInfo(static_cast<uint32_t>(GGUFTensorType::FLOAT16),
                                                &type_info));
  EXPECT_EQ(type_info.block_size_bytes, sizeof(GGUFFloat16));

  ASSERT_TRUE(
      GGUFSerializer::GetTensorTypeInfo(static_cast<uint32_t>(GGUFTensorType::INT8), &type_info));
  EXPECT_EQ(type_info.block_size_bytes, sizeof(int8_t));

  ASSERT_TRUE(
      GGUFSerializer::GetTensorTypeInfo(static_cast<uint32_t>(GGUFTensorType::NVFP4), &type_info));
  EXPECT_EQ(type_info.block_element_count, 64U);
  EXPECT_EQ(type_info.block_size_bytes, sizeof(GGUFNVFP4Block));
  EXPECT_TRUE(type_info.is_quantized);

  ASSERT_TRUE(GGUFSerializer::GetTensorTypeInfo(static_cast<uint32_t>(GGUFTensorType::FP8_E4M3),
                                                &type_info));
  EXPECT_EQ(type_info.block_size_bytes, sizeof(GGUFFloat8E4M3));
  EXPECT_TRUE(type_info.is_extension);

  EXPECT_FALSE(GGUFSerializer::GetTensorTypeInfo(UINT32_MAX, &type_info));
  EXPECT_FALSE(GGUFSerializer::IsGGUFFile("model/qwen3_vl/README.md"));
}

} // namespace
} // namespace util
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
