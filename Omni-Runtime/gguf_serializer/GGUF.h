#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace omni_runtime {
namespace util {

inline constexpr std::array<char, 4> kGGUFMagic = {'G', 'G', 'U', 'F'};
inline constexpr uint32_t kGGUFVersion2 = 2U;
inline constexpr uint32_t kGGUFVersion3 = 3U;
inline constexpr uint32_t kGGUFDefaultAlignmentBytes = 32U;
inline constexpr uint32_t kGGUFMaxTensorDimensions = 8U;

enum class GGUFValueType : uint32_t {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
};

// Values 0-42 follow ggml. FP8 values are extension identifiers; files using
// another vendor identifier remain readable through the raw type id and
// byte-oriented tensor API.
enum class GGUFTensorType : uint32_t {
  FLOAT32 = 0,
  FLOAT16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  INT8 = 24,
  INT16 = 25,
  INT32 = 26,
  INT64 = 27,
  FLOAT64 = 28,
  IQ1_M = 29,
  BFLOAT16 = 30,
  TQ1_0 = 34,
  TQ2_0 = 35,
  MXFP4 = 39,
  NVFP4 = 40,
  Q1_0 = 41,
  Q2_0 = 42,
  FP8_E4M3 = 43,
  FP8_E5M2 = 44,
};

struct GGUFFloat16 final {
  uint16_t bits = 0U;
};

struct GGUFBFloat16 final {
  uint16_t bits = 0U;
};

struct GGUFFloat8E4M3 final {
  uint8_t bits = 0U;
};

struct GGUFFloat8E5M2 final {
  uint8_t bits = 0U;
};

template <size_t SizeBytes> struct GGUFStorageBlock final {
  std::array<uint8_t, SizeBytes> bytes = {};
};

using GGUFNVFP4Block = GGUFStorageBlock<36U>;

using GGUFScalarValue = std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float,
                                     bool, std::string, uint64_t, int64_t, double>;

struct GGUFValue final {
  GGUFValueType type = GGUFValueType::UINT8;
  GGUFScalarValue scalar = uint8_t{0U};
  std::vector<GGUFValue> array;

  template <typename T> bool Get(T *const value) const {
    if (value == nullptr) {
      return false;
    }
    const T *const typed_value = std::get_if<T>(&scalar);
    if (typed_value == nullptr || type == GGUFValueType::ARRAY) {
      return false;
    }
    *value = *typed_value;
    return true;
  }

  template <typename T> bool GetArray(std::vector<T> *const values) const {
    if (values == nullptr || type != GGUFValueType::ARRAY) {
      return false;
    }
    values->clear();
    values->reserve(array.size());
    for (const GGUFValue &item : array) {
      T value;
      if (!item.Get(&value)) {
        values->clear();
        return false;
      }
      values->emplace_back(std::move(value));
    }
    return true;
  }
};

struct GGUFHeader final {
  uint32_t version = 0U;
  uint64_t tensor_count = 0U;
  uint64_t metadata_count = 0U;
};

struct GGUFMetadata final {
  std::string key;
  GGUFValue value;
};

struct GGUFTensorTypeInfo final {
  uint32_t type_id = 0U;
  std::string name;
  uint64_t block_element_count = 0U;
  uint64_t block_size_bytes = 0U;
  bool is_quantized = false;
  bool is_extension = false;
};

struct GGUFTensorInfo final {
  std::string name;
  std::vector<uint64_t> dimensions;
  uint32_t type_id = 0U;
  uint64_t element_count = 0U;
  uint64_t relative_offset_bytes = 0U;
  uint64_t data_offset_bytes = 0U;
  uint64_t storage_size_bytes = 0U;
  uint64_t storage_span_bytes = 0U;
  bool has_known_storage_size = false;
};

struct GGUFModelInfo final {
  std::string file_path;
  uint64_t file_size_bytes = 0U;
  uint64_t data_offset_bytes = 0U;
  uint32_t alignment_bytes = kGGUFDefaultAlignmentBytes;
  GGUFHeader header;
  std::vector<GGUFMetadata> metadata;
  std::vector<GGUFTensorInfo> tensors;
};

} // namespace util
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
