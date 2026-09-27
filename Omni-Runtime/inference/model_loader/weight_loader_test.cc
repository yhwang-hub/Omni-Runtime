#include "Omni-Runtime/inference/model_loader/weight_loader.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "gtest/gtest.h"

namespace omni_runtime {
namespace inference {
namespace {

class WeightLoaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *const root = std::getenv("TEST_SRCDIR");
    const auto directory =
        root == nullptr ? std::filesystem::path() : std::filesystem::path(root) / "_main";
    ASSERT_TRUE(
        serializer_.Parse((directory / "model/qwen3_vl/Qwen3VL-8B-Instruct-F16.gguf").string()))
        << serializer_.error_message();
  }

  util::GGUFSerializer serializer_;
};

TEST_F(WeightLoaderTest, PreservesRawEncodingAndExactFloatConversionOnBothBackends) {
  const std::string name = "output_norm.weight";
  const auto *const source = serializer_.FindTensor(name);
  ASSERT_NE(source, nullptr);
  ASSERT_EQ(source->type_id, static_cast<uint32_t>(util::GGUFTensorType::FLOAT32));
  std::vector<float> expected;
  ASSERT_TRUE(serializer_.ReadTensorData<float>(name, 0U, source->element_count, &expected));
  for (const std::string backend_name : {"cuda", "opencl"}) {
    SCOPED_TRACE(backend_name);
    const auto device = backend::CreateComputeBackend(backend_name);
    ASSERT_NE(device, nullptr);
    WeightLoader loader;
    memory::BufferPtr encoded;
    ASSERT_TRUE(loader.LoadEncoded(serializer_, name, device, &encoded)) << loader.error_message();
    std::vector<float> raw;
    ASSERT_TRUE(encoded->CopyDataToVector(&raw));
    EXPECT_EQ(raw, expected);
    for (const auto type :
         {ModelWeightType::kFloat32, ModelWeightType::kFloat16, ModelWeightType::kBFloat16}) {
      memory::TensorPtr tensor;
      ASSERT_TRUE(loader.LoadTensor(serializer_, name, type, device, &tensor))
          << loader.error_message();
      EXPECT_EQ(tensor->element_count(), expected.size());
      if (type == ModelWeightType::kFloat32) {
        std::vector<float> actual(expected.size());
        ASSERT_TRUE(tensor->CopyToHost(actual.data(), actual.size() * sizeof(float)));
        EXPECT_EQ(actual, expected);
        continue;
      }
      std::vector<uint16_t> actual(expected.size());
      ASSERT_TRUE(tensor->CopyToHost(actual.data(), actual.size() * sizeof(uint16_t)));
      for (size_t index = 0U; index < expected.size(); ++index) {
        uint16_t bits = 0U;
        if (type == ModelWeightType::kFloat16) {
          const _Float16 value = static_cast<_Float16>(expected[index]);
          std::memcpy(&bits, &value, sizeof(bits));
        } else {
          uint32_t value = 0U;
          std::memcpy(&value, &expected[index], sizeof(value));
          bits = static_cast<uint16_t>((value + 0x7FFFU + ((value >> 16U) & 1U)) >> 16U);
        }
        EXPECT_EQ(actual[index], bits) << index;
      }
    }
  }
}

TEST_F(WeightLoaderTest, RejectsInvalidInputsWithoutReplacingExistingStorageAndRecovers) {
  const auto device = backend::CreateComputeBackend("cuda");
  ASSERT_NE(device, nullptr);
  WeightLoader loader;
  memory::TensorPtr tensor;
  ASSERT_TRUE(loader.LoadTensor(serializer_, "output_norm.weight", ModelWeightType::kFloat32,
                                device, &tensor));
  const auto original = tensor;
  EXPECT_FALSE(
      loader.LoadTensor(serializer_, "missing", ModelWeightType::kFloat32, device, &tensor));
  EXPECT_FALSE(loader.error_message().empty());
  EXPECT_FALSE(loader.LoadTensor(serializer_, "output_norm.weight", ModelWeightType::kAuto, device,
                                 &tensor));
  EXPECT_FALSE(loader.LoadTensor(serializer_, "output_norm.weight", ModelWeightType::kFloat32,
                                 nullptr, &tensor));
  EXPECT_FALSE(loader.LoadTensor(serializer_, "output_norm.weight", ModelWeightType::kFloat32,
                                 device, nullptr));
  EXPECT_EQ(tensor, original);
  memory::BufferPtr buffer = original->buffer();
  EXPECT_FALSE(loader.LoadEncoded(serializer_, "missing", device, &buffer));
  EXPECT_FALSE(loader.LoadEncoded(serializer_, "output_norm.weight", nullptr, &buffer));
  EXPECT_FALSE(loader.LoadEncoded(serializer_, "output_norm.weight", device, nullptr));
  EXPECT_EQ(buffer, original->buffer());
  ASSERT_TRUE(loader.LoadEncoded(serializer_, "output_norm.weight", device, &buffer));
  EXPECT_TRUE(loader.error_message().empty());
}

} // namespace
} // namespace inference
} // namespace omni_runtime
