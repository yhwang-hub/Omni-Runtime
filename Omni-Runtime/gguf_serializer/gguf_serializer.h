#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Omni-Runtime/gguf_serializer/GGUF.h"

namespace omni_runtime {
namespace util {

template <typename T> struct GGUFTensorStorageTraits {
  static constexpr bool kIsSupported = false;
  static constexpr uint32_t kTypeId = 0U;
};

#define OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(storage_type, tensor_type)                         \
  template <> struct GGUFTensorStorageTraits<storage_type> {                                       \
    static constexpr bool kIsSupported = true;                                                     \
    static constexpr uint32_t kTypeId = static_cast<uint32_t>(GGUFTensorType::tensor_type);        \
  }

OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(float, FLOAT32);
OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(GGUFFloat16, FLOAT16);
OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(int8_t, INT8);
OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(int16_t, INT16);
OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(int32_t, INT32);
OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(int64_t, INT64);
OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(double, FLOAT64);
OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(GGUFBFloat16, BFLOAT16);
OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(GGUFFloat8E4M3, FP8_E4M3);
OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(GGUFFloat8E5M2, FP8_E5M2);
OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS(GGUFNVFP4Block, NVFP4);

#undef OMNI_RUNTIME_DEFINE_GGUF_STORAGE_TRAITS

class GGUFSerializer final {
public:
  GGUFSerializer();
  ~GGUFSerializer();
  GGUFSerializer(const GGUFSerializer &) = delete;
  GGUFSerializer &operator=(const GGUFSerializer &) = delete;
  GGUFSerializer(GGUFSerializer &&) = delete;
  GGUFSerializer &operator=(GGUFSerializer &&) = delete;

public:
  bool Parse(const std::string &file_path);
  bool IsParsed() const;
  bool HasMetadata(const std::string &key) const;
  bool HasTensor(const std::string &name) const;

  const GGUFModelInfo &model() const;
  const std::string &error_message() const;
  const GGUFMetadata *FindMetadata(const std::string &key) const;
  const GGUFTensorInfo *FindTensor(const std::string &name) const;

  template <typename T> bool GetMetadata(const std::string &key, T *const value) const {
    if (value == nullptr) {
      return false;
    }
    const GGUFMetadata *const metadata = FindMetadata(key);
    return metadata != nullptr && metadata->value.Get(value);
  }

  template <typename T>
  bool GetMetadataArray(const std::string &key, std::vector<T> *const values) const {
    if (values == nullptr) {
      return false;
    }
    const GGUFMetadata *const metadata = FindMetadata(key);
    return metadata != nullptr && metadata->value.GetArray(values);
  }

  bool ReadTensorBytes(const std::string &name, const uint64_t offset_bytes,
                       const size_t size_bytes, std::vector<uint8_t> *const data) const;

  template <typename T>
  bool ReadTensorData(const std::string &name, const uint64_t storage_offset,
                      const size_t storage_count, std::vector<T> *const data) const {
    if (data == nullptr || !GGUFTensorStorageTraits<T>::kIsSupported) {
      return false;
    }
    const GGUFTensorInfo *const tensor = FindTensor(name);
    if (tensor == nullptr || tensor->type_id != GGUFTensorStorageTraits<T>::kTypeId) {
      return false;
    }
    if (storage_count > static_cast<size_t>(UINT64_MAX / sizeof(T))) {
      return false;
    }
    const uint64_t size_bytes = static_cast<uint64_t>(storage_count) * sizeof(T);
    if (storage_offset > UINT64_MAX / sizeof(T)) {
      return false;
    }
    const uint64_t offset_bytes = storage_offset * sizeof(T);
    if (size_bytes > static_cast<uint64_t>(SIZE_MAX)) {
      return false;
    }
    std::vector<uint8_t> bytes;
    if (!ReadTensorBytes(name, offset_bytes, static_cast<size_t>(size_bytes), &bytes)) {
      return false;
    }
    data->resize(storage_count);
    if (!bytes.empty()) {
      std::memcpy(data->data(), bytes.data(), bytes.size());
    }
    return true;
  }

  static bool IsGGUFFile(const std::string &file_path);
  static bool GetTensorTypeInfo(const uint32_t type_id, GGUFTensorTypeInfo *const type_info);

private:
  class Reader;

  bool ParseHeader();
  bool ParseMetadata();
  bool ParseMetadataValue(const GGUFValueType type, const uint32_t depth, GGUFValue *const value);
  bool ParseTensorInfo();
  bool FinalizeTensorLayout();
  bool ReadString(std::string *const value);
  bool SetError(const std::string &message);
  bool Reset();

private:
  std::unique_ptr<Reader> reader_;
  GGUFModelInfo model_;
  std::unordered_map<std::string, size_t> metadata_index_;
  std::unordered_map<std::string, size_t> tensor_index_;
  std::string error_message_;
  bool is_parsed_ = false;
};

} // namespace util
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
