#include "Omni-Runtime/inference/model_loader/weight_loader.h"

#include <cmath>
#include <cstring>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace {

float HalfToFloat(const uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits >> 15U) << 31U;
  const uint32_t exponent = static_cast<uint32_t>((bits >> 10U) & 0x1FU);
  const uint32_t fraction = static_cast<uint32_t>(bits & 0x3FFU);
  uint32_t value = 0U;
  if (exponent == 0U) {
    return std::ldexp(static_cast<float>(fraction), -24) * (sign == 0U ? 1.0F : -1.0F);
  } else if (exponent == 0x1FU) {
    value = 0x7F800000U | (fraction << 13U);
  } else {
    value = ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  float output = 0.0F;
  std::memcpy(&output, &value, sizeof(output));
  return sign == 0U ? output : -output;
}

uint16_t FloatToBFloat16(const float value) {
  uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>((bits + rounding) >> 16U);
}

uint16_t FloatToHalf(const float value) {
  const _Float16 half = static_cast<_Float16>(value);
  uint16_t bits = 0U;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

template <typename T>
bool Upload(const std::shared_ptr<backend::Backend> &backend,
            const std::vector<uint64_t> &dimensions, const std::vector<T> &values,
            const memory::DataType type, memory::TensorPtr *const tensor) {
  OMNI_RETURN_VAL_IF(tensor == nullptr || backend == nullptr || values.empty(), false);
  const std::vector<size_t> shape(dimensions.begin(), dimensions.end());
  auto output = memory::Tensor::Create(backend, shape, type, backend::MemoryKind::kDevice);
  OMNI_RETURN_VAL_IF(
      output == nullptr || !output->CopyFromHost(values.data(), values.size() * sizeof(T)), false);
  *tensor = std::move(output);
  return true;
}

bool ConvertTensor(const util::GGUFSerializer &serializer, const util::GGUFTensorInfo &source,
                   const ModelWeightType compute_type,
                   const std::shared_ptr<backend::Backend> &backend,
                   memory::TensorPtr *const tensor) {
  OMNI_RETURN_VAL_IF(tensor == nullptr || backend == nullptr, false);
  const auto type = static_cast<util::GGUFTensorType>(source.type_id);
  const size_t count = static_cast<size_t>(source.element_count);
  if (compute_type == ModelWeightType::kFloat32 && type == util::GGUFTensorType::FLOAT32) {
    std::vector<float> values;
    return serializer.ReadTensorData<float>(source.name, 0U, count, &values) &&
           Upload(backend, source.dimensions, values, memory::DataType::kFloat32, tensor);
  }
  if (compute_type == ModelWeightType::kFloat16 && type == util::GGUFTensorType::FLOAT16) {
    std::vector<util::GGUFFloat16> values;
    return serializer.ReadTensorData<util::GGUFFloat16>(source.name, 0U, count, &values) &&
           Upload(backend, source.dimensions, values, memory::DataType::kFloat16, tensor);
  }
  if (compute_type == ModelWeightType::kBFloat16 && type == util::GGUFTensorType::BFLOAT16) {
    std::vector<util::GGUFBFloat16> values;
    OMNI_RETURN_VAL_IF(
        !serializer.ReadTensorData<util::GGUFBFloat16>(source.name, 0U, count, &values), false);
    std::vector<uint16_t> converted;
    converted.reserve(values.size());
    for (const auto value : values) {
      converted.emplace_back(value.bits);
    }
    return Upload(backend, source.dimensions, converted, memory::DataType::kBFloat16, tensor);
  }
  if (type == util::GGUFTensorType::FLOAT16 && compute_type == ModelWeightType::kFloat32) {
    std::vector<util::GGUFFloat16> values;
    OMNI_RETURN_VAL_IF(
        !serializer.ReadTensorData<util::GGUFFloat16>(source.name, 0U, count, &values), false);
    std::vector<float> converted;
    converted.reserve(values.size());
    for (const auto value : values) {
      converted.emplace_back(HalfToFloat(value.bits));
    }
    return Upload(backend, source.dimensions, converted, memory::DataType::kFloat32, tensor);
  }
  if (type == util::GGUFTensorType::FLOAT32 &&
      (compute_type == ModelWeightType::kFloat16 || compute_type == ModelWeightType::kBFloat16)) {
    std::vector<float> values;
    OMNI_RETURN_VAL_IF(!serializer.ReadTensorData<float>(source.name, 0U, count, &values), false);
    std::vector<uint16_t> converted;
    converted.reserve(values.size());
    for (const float value : values) {
      converted.emplace_back(compute_type == ModelWeightType::kFloat16 ? FloatToHalf(value)
                                                                       : FloatToBFloat16(value));
    }
    return Upload(backend, source.dimensions, converted,
                  compute_type == ModelWeightType::kFloat16 ? memory::DataType::kFloat16
                                                            : memory::DataType::kBFloat16,
                  tensor);
  }
  return false;
}

} // namespace

bool WeightLoader::LoadTensor(const util::GGUFSerializer &serializer, const std::string &name,
                              const ModelWeightType compute_type,
                              const std::shared_ptr<backend::Backend> &backend,
                              memory::TensorPtr *const tensor) {
  OMNI_RETURN_VAL_IF(tensor == nullptr || backend == nullptr,
                     SetError("weight output or backend is null"));
  error_message_.clear();
  const auto *const source = serializer.FindTensor(name);
  OMNI_RETURN_VAL_IF(source == nullptr, SetError("missing GGUF tensor: " + name));
  OMNI_RETURN_VAL_IF(!ConvertTensor(serializer, *source, compute_type, backend, tensor),
                     SetError("cannot read, convert or upload GGUF tensor: " + name));
  return true;
}

bool WeightLoader::LoadEncoded(const util::GGUFSerializer &serializer, const std::string &name,
                               const std::shared_ptr<backend::Backend> &backend,
                               memory::BufferPtr *const buffer) {
  OMNI_RETURN_VAL_IF(buffer == nullptr || backend == nullptr,
                     SetError("weight output or backend is null"));
  error_message_.clear();
  const auto *const source = serializer.FindTensor(name);
  OMNI_RETURN_VAL_IF(source == nullptr || !source->has_known_storage_size,
                     SetError("missing GGUF tensor or unknown storage encoding: " + name));
  std::vector<uint8_t> bytes;
  OMNI_RETURN_VAL_IF(!serializer.ReadTensorBytes(name, 0U, source->storage_size_bytes, &bytes),
                     SetError("cannot read GGUF tensor: " + name));
  auto output = memory::Buffer::CreateDeviceBufferFromVector(backend, bytes);
  OMNI_RETURN_VAL_IF(output == nullptr, SetError("cannot upload GGUF tensor: " + name));
  *buffer = std::move(output);
  return true;
}

bool WeightLoader::SetError(const std::string &message) {
  error_message_ = message;
  return false;
}

} // namespace inference
} // namespace omni_runtime
