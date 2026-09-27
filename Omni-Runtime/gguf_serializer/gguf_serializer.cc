#include "Omni-Runtime/gguf_serializer/gguf_serializer.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <type_traits>
#include <utility>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace util {

namespace {

constexpr uint32_t kMaxMetadataDepth = 4U;
constexpr uint64_t kMinimumMetadataBytes = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint8_t);
constexpr uint64_t kMinimumTensorInfoBytes =
    sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t);

bool IsPowerOfTwo(const uint32_t value) {
  return value != 0U && (value & (value - 1U)) == 0U;
}

bool AlignUp(const uint64_t source, const uint64_t alignment, uint64_t *const aligned_value) {
  OMNI_RETURN_VAL_IF(aligned_value == nullptr || alignment == 0U, false);
  const uint64_t remainder = source % alignment;
  if (remainder == 0U) {
    *aligned_value = source;
    return true;
  }
  const uint64_t padding = alignment - remainder;
  OMNI_RETURN_VAL_IF(source > UINT64_MAX - padding, false);
  *aligned_value = source + padding;
  return true;
}

bool Multiply(const uint64_t lhs, const uint64_t rhs, uint64_t *const product) {
  OMNI_RETURN_VAL_IF(product == nullptr, false);
  OMNI_RETURN_VAL_IF(lhs != 0U && rhs > UINT64_MAX / lhs, false);
  *product = lhs * rhs;
  return true;
}

bool SetTensorTypeInfo(const uint32_t type_id, const char *const name,
                       const uint64_t block_element_count, const uint64_t block_size_bytes,
                       const bool is_quantized, const bool is_extension,
                       GGUFTensorTypeInfo *const type_info) {
  OMNI_RETURN_VAL_IF(name == nullptr || type_info == nullptr, false);
  type_info->type_id = type_id;
  type_info->name = name;
  type_info->block_element_count = block_element_count;
  type_info->block_size_bytes = block_size_bytes;
  type_info->is_quantized = is_quantized;
  type_info->is_extension = is_extension;
  return true;
}

} // namespace

class GGUFSerializer::Reader final {
public:
  explicit Reader(const std::string &file_path) : file_path_(file_path) {
  }

  bool Open() {
    stream_.open(file_path_, std::ios::binary | std::ios::ate);
    OMNI_RETURN_VAL_IF(!stream_.is_open(), false);
    const std::ifstream::pos_type file_end = stream_.tellg();
    OMNI_RETURN_VAL_IF(file_end < 0, false);
    file_size_bytes_ = static_cast<uint64_t>(file_end);
    stream_.seekg(0, std::ios::beg);
    offset_bytes_ = 0U;
    return stream_.good();
  }

  template <typename T> bool Read(T *const output) {
    static_assert(std::is_trivially_copyable<T>::value, "GGUF values must be trivially copyable");
    OMNI_RETURN_VAL_IF(output == nullptr, false);
    std::array<uint8_t, sizeof(T)> bytes = {};
    OMNI_RETURN_VAL_IF(!ReadBytes(bytes.data(), bytes.size()), false);
    const uint16_t endian_probe = 1U;
    const bool is_little_endian = *reinterpret_cast<const uint8_t *>(&endian_probe) == 1U;
    if (!is_little_endian && sizeof(T) > 1U) {
      std::reverse(bytes.begin(), bytes.end());
    }
    std::memcpy(output, bytes.data(), sizeof(T));
    return true;
  }

  bool ReadBytes(void *const data, const size_t size_bytes) {
    OMNI_RETURN_VAL_IF(data == nullptr && size_bytes != 0U, false);
    OMNI_RETURN_VAL_IF(size_bytes > remaining_bytes(), false);
    uint8_t *output = static_cast<uint8_t *>(data);
    size_t read_size_bytes = 0U;
    while (read_size_bytes < size_bytes) {
      const size_t remaining_size_bytes = size_bytes - read_size_bytes;
      const size_t chunk_size_bytes = std::min(
          remaining_size_bytes, static_cast<size_t>(std::numeric_limits<std::streamsize>::max()));
      stream_.read(reinterpret_cast<char *>(output + read_size_bytes),
                   static_cast<std::streamsize>(chunk_size_bytes));
      OMNI_RETURN_VAL_IF(!stream_ || static_cast<size_t>(stream_.gcount()) != chunk_size_bytes, false);
      read_size_bytes += chunk_size_bytes;
    }
    offset_bytes_ += size_bytes;
    return true;
  }

  bool Seek(const uint64_t offset_bytes) {
    OMNI_RETURN_VAL_IF(offset_bytes > file_size_bytes_, false);
    OMNI_RETURN_VAL_IF(offset_bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()),
                  false);
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(offset_bytes), std::ios::beg);
    OMNI_RETURN_VAL_IF(!stream_.good(), false);
    offset_bytes_ = offset_bytes;
    return true;
  }

  bool ReadAt(const uint64_t offset_bytes, const size_t size_bytes, void *const data) const {
    OMNI_RETURN_VAL_IF(data == nullptr && size_bytes != 0U, false);
    OMNI_RETURN_VAL_IF(offset_bytes > file_size_bytes_ || size_bytes > file_size_bytes_ - offset_bytes,
                  false);
    std::ifstream input(file_path_, std::ios::binary);
    OMNI_RETURN_VAL_IF(!input.is_open(), false);
    OMNI_RETURN_VAL_IF(offset_bytes > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()),
                  false);
    input.seekg(static_cast<std::streamoff>(offset_bytes), std::ios::beg);
    OMNI_RETURN_VAL_IF(!input.good(), false);

    uint8_t *output = static_cast<uint8_t *>(data);
    size_t read_size_bytes = 0U;
    while (read_size_bytes < size_bytes) {
      const size_t remaining_size_bytes = size_bytes - read_size_bytes;
      const size_t chunk_size_bytes = std::min(
          remaining_size_bytes, static_cast<size_t>(std::numeric_limits<std::streamsize>::max()));
      input.read(reinterpret_cast<char *>(output + read_size_bytes),
                 static_cast<std::streamsize>(chunk_size_bytes));
      OMNI_RETURN_VAL_IF(!input || static_cast<size_t>(input.gcount()) != chunk_size_bytes, false);
      read_size_bytes += chunk_size_bytes;
    }
    return true;
  }

  uint64_t offset_bytes() const {
    return offset_bytes_;
  }

  uint64_t file_size_bytes() const {
    return file_size_bytes_;
  }

  uint64_t remaining_bytes() const {
    return file_size_bytes_ - offset_bytes_;
  }

private:
  std::string file_path_;
  std::ifstream stream_;
  uint64_t file_size_bytes_ = 0U;
  uint64_t offset_bytes_ = 0U;
};

GGUFSerializer::GGUFSerializer() = default;

GGUFSerializer::~GGUFSerializer() = default;

bool GGUFSerializer::Parse(const std::string &file_path) {
  OMNI_RETURN_VAL_IF(!Reset(), false);
  OMNI_RETURN_VAL_IF(file_path.empty(), SetError("GGUF file path is empty"));
  reader_ = std::make_unique<Reader>(file_path);
  OMNI_RETURN_VAL_IF(reader_ == nullptr || !reader_->Open(),
                SetError("failed to open GGUF file: " + file_path));
  model_.file_path = file_path;
  model_.file_size_bytes = reader_->file_size_bytes();
  OMNI_RETURN_VAL_IF(!ParseHeader(), false);
  OMNI_RETURN_VAL_IF(!ParseMetadata(), false);
  OMNI_RETURN_VAL_IF(!ParseTensorInfo(), false);
  OMNI_RETURN_VAL_IF(!FinalizeTensorLayout(), false);
  is_parsed_ = true;
  return true;
}

bool GGUFSerializer::ParseHeader() {
  std::array<char, kGGUFMagic.size()> magic = {};
  OMNI_RETURN_VAL_IF(!reader_->ReadBytes(magic.data(), magic.size()),
                SetError("failed to read GGUF magic"));
  OMNI_RETURN_VAL_IF(!std::equal(magic.begin(), magic.end(), kGGUFMagic.begin()),
                SetError("invalid GGUF magic"));
  OMNI_RETURN_VAL_IF(!reader_->Read(&model_.header.version), SetError("failed to read GGUF version"));
  OMNI_RETURN_VAL_IF(model_.header.version != kGGUFVersion2 && model_.header.version != kGGUFVersion3,
                SetError("unsupported GGUF version: " + std::to_string(model_.header.version)));
  OMNI_RETURN_VAL_IF(!reader_->Read(&model_.header.tensor_count),
                SetError("failed to read GGUF tensor count"));
  OMNI_RETURN_VAL_IF(!reader_->Read(&model_.header.metadata_count),
                SetError("failed to read GGUF metadata count"));
  OMNI_RETURN_VAL_IF(model_.header.metadata_count > reader_->remaining_bytes() / kMinimumMetadataBytes,
                SetError("invalid GGUF metadata count"));
  return true;
}

bool GGUFSerializer::ParseMetadata() {
  OMNI_RETURN_VAL_IF(model_.header.metadata_count > static_cast<uint64_t>(SIZE_MAX),
                SetError("GGUF metadata count exceeds host limits"));
  model_.metadata.reserve(static_cast<size_t>(model_.header.metadata_count));
  metadata_index_.reserve(static_cast<size_t>(model_.header.metadata_count));
  for (uint64_t index = 0U; index < model_.header.metadata_count; ++index) {
    GGUFMetadata metadata;
    OMNI_RETURN_VAL_IF(!ReadString(&metadata.key), SetError("failed to read GGUF metadata key"));
    OMNI_RETURN_VAL_IF(metadata.key.empty(), SetError("GGUF metadata key is empty"));
    OMNI_RETURN_VAL_IF(metadata_index_.find(metadata.key) != metadata_index_.end(),
                  SetError("duplicated GGUF metadata key: " + metadata.key));

    uint32_t type_id = 0U;
    OMNI_RETURN_VAL_IF(!reader_->Read(&type_id),
                  SetError("failed to read GGUF metadata type: " + metadata.key));
    OMNI_RETURN_VAL_IF(type_id > static_cast<uint32_t>(GGUFValueType::FLOAT64),
                  SetError("unsupported GGUF metadata type: " + std::to_string(type_id)));
    OMNI_RETURN_VAL_IF(!ParseMetadataValue(static_cast<GGUFValueType>(type_id), 0U, &metadata.value),
                  false);

    metadata_index_.emplace(metadata.key, model_.metadata.size());
    model_.metadata.emplace_back(std::move(metadata));
  }
  return true;
}

bool GGUFSerializer::ParseMetadataValue(const GGUFValueType type, const uint32_t depth,
                                        GGUFValue *const output) {
  OMNI_RETURN_VAL_IF(output == nullptr, SetError("GGUF metadata output is null"));
  OMNI_RETURN_VAL_IF(depth > kMaxMetadataDepth, SetError("GGUF metadata nesting is too deep"));
  output->type = type;
  switch (type) {
  case GGUFValueType::UINT8: {
    uint8_t scalar = 0U;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF uint8 metadata"));
    output->scalar = scalar;
    return true;
  }
  case GGUFValueType::INT8: {
    int8_t scalar = 0;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF int8 metadata"));
    output->scalar = scalar;
    return true;
  }
  case GGUFValueType::UINT16: {
    uint16_t scalar = 0U;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF uint16 metadata"));
    output->scalar = scalar;
    return true;
  }
  case GGUFValueType::INT16: {
    int16_t scalar = 0;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF int16 metadata"));
    output->scalar = scalar;
    return true;
  }
  case GGUFValueType::UINT32: {
    uint32_t scalar = 0U;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF uint32 metadata"));
    output->scalar = scalar;
    return true;
  }
  case GGUFValueType::INT32: {
    int32_t scalar = 0;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF int32 metadata"));
    output->scalar = scalar;
    return true;
  }
  case GGUFValueType::FLOAT32: {
    float scalar = 0.0f;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF float32 metadata"));
    output->scalar = scalar;
    return true;
  }
  case GGUFValueType::BOOL: {
    uint8_t scalar = 0U;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF bool metadata"));
    OMNI_RETURN_VAL_IF(scalar > 1U, SetError("invalid GGUF bool metadata"));
    output->scalar = scalar != 0U;
    return true;
  }
  case GGUFValueType::STRING: {
    std::string scalar;
    OMNI_RETURN_VAL_IF(!ReadString(&scalar), SetError("failed to read GGUF string metadata"));
    output->scalar = std::move(scalar);
    return true;
  }
  case GGUFValueType::ARRAY: {
    uint32_t item_type_id = 0U;
    uint64_t item_count = 0U;
    OMNI_RETURN_VAL_IF(!reader_->Read(&item_type_id) || !reader_->Read(&item_count),
                  SetError("failed to read GGUF array metadata"));
    OMNI_RETURN_VAL_IF(item_type_id > static_cast<uint32_t>(GGUFValueType::FLOAT64),
                  SetError("unsupported GGUF array item type: " + std::to_string(item_type_id)));
    OMNI_RETURN_VAL_IF(item_count > static_cast<uint64_t>(SIZE_MAX),
                  SetError("GGUF array exceeds host limits"));
    output->array.reserve(static_cast<size_t>(item_count));
    for (uint64_t index = 0U; index < item_count; ++index) {
      GGUFValue item;
      OMNI_RETURN_VAL_IF(
          !ParseMetadataValue(static_cast<GGUFValueType>(item_type_id), depth + 1U, &item), false);
      output->array.emplace_back(std::move(item));
    }
    return true;
  }
  case GGUFValueType::UINT64: {
    uint64_t scalar = 0U;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF uint64 metadata"));
    output->scalar = scalar;
    return true;
  }
  case GGUFValueType::INT64: {
    int64_t scalar = 0;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF int64 metadata"));
    output->scalar = scalar;
    return true;
  }
  case GGUFValueType::FLOAT64: {
    double scalar = 0.0;
    OMNI_RETURN_VAL_IF(!reader_->Read(&scalar), SetError("failed to read GGUF float64 metadata"));
    output->scalar = scalar;
    return true;
  }
  default:
    return SetError("unsupported GGUF metadata value");
  }
}

bool GGUFSerializer::ParseTensorInfo() {
  OMNI_RETURN_VAL_IF(model_.header.tensor_count > reader_->remaining_bytes() / kMinimumTensorInfoBytes,
                SetError("invalid GGUF tensor count"));
  OMNI_RETURN_VAL_IF(model_.header.tensor_count > static_cast<uint64_t>(SIZE_MAX),
                SetError("GGUF tensor count exceeds host limits"));
  model_.tensors.reserve(static_cast<size_t>(model_.header.tensor_count));
  tensor_index_.reserve(static_cast<size_t>(model_.header.tensor_count));

  for (uint64_t index = 0U; index < model_.header.tensor_count; ++index) {
    GGUFTensorInfo tensor;
    OMNI_RETURN_VAL_IF(!ReadString(&tensor.name), SetError("failed to read GGUF tensor name"));
    OMNI_RETURN_VAL_IF(tensor.name.empty(), SetError("GGUF tensor name is empty"));
    OMNI_RETURN_VAL_IF(tensor_index_.find(tensor.name) != tensor_index_.end(),
                  SetError("duplicated GGUF tensor name: " + tensor.name));

    uint32_t dimension_count = 0U;
    OMNI_RETURN_VAL_IF(!reader_->Read(&dimension_count),
                  SetError("failed to read GGUF tensor dimension count: " + tensor.name));
    OMNI_RETURN_VAL_IF(dimension_count == 0U || dimension_count > kGGUFMaxTensorDimensions,
                  SetError("invalid GGUF tensor dimension count: " + tensor.name));
    tensor.dimensions.resize(dimension_count);
    tensor.element_count = 1U;
    for (uint32_t dimension_index = 0U; dimension_index < dimension_count; ++dimension_index) {
      uint64_t dimension = 0U;
      OMNI_RETURN_VAL_IF(!reader_->Read(&dimension),
                    SetError("failed to read GGUF tensor dimension: " + tensor.name));
      OMNI_RETURN_VAL_IF(dimension == 0U,
                    SetError("GGUF tensor dimension must be positive: " + tensor.name));
      OMNI_RETURN_VAL_IF(!Multiply(tensor.element_count, dimension, &tensor.element_count),
                    SetError("GGUF tensor element count overflow: " + tensor.name));
      tensor.dimensions[dimension_index] = dimension;
    }
    OMNI_RETURN_VAL_IF(!reader_->Read(&tensor.type_id) || !reader_->Read(&tensor.relative_offset_bytes),
                  SetError("failed to read GGUF tensor descriptor: " + tensor.name));
    tensor_index_.emplace(tensor.name, model_.tensors.size());
    model_.tensors.emplace_back(std::move(tensor));
  }
  return true;
}

bool GGUFSerializer::FinalizeTensorLayout() {
  uint32_t alignment_bytes = kGGUFDefaultAlignmentBytes;
  const GGUFMetadata *const alignment_metadata =
      HasMetadata("general.alignment") ? FindMetadata("general.alignment") : nullptr;
  if (alignment_metadata != nullptr) {
    OMNI_RETURN_VAL_IF(!alignment_metadata->value.Get(&alignment_bytes),
                  SetError("general.alignment must be uint32"));
  }
  OMNI_RETURN_VAL_IF(!IsPowerOfTwo(alignment_bytes), SetError("GGUF alignment is invalid"));
  model_.alignment_bytes = alignment_bytes;
  OMNI_RETURN_VAL_IF(!AlignUp(reader_->offset_bytes(), alignment_bytes, &model_.data_offset_bytes),
                SetError("GGUF data offset overflow"));
  OMNI_RETURN_VAL_IF(model_.data_offset_bytes > model_.file_size_bytes,
                SetError("GGUF data offset exceeds file size"));

  std::vector<size_t> ordered_indices(model_.tensors.size());
  for (size_t index = 0U; index < ordered_indices.size(); ++index) {
    ordered_indices[index] = index;
  }
  std::sort(ordered_indices.begin(), ordered_indices.end(),
            [this](const size_t lhs, const size_t rhs) {
              return model_.tensors[lhs].relative_offset_bytes <
                     model_.tensors[rhs].relative_offset_bytes;
            });

  for (size_t order_index = 0U; order_index < ordered_indices.size(); ++order_index) {
    GGUFTensorInfo &tensor = model_.tensors[ordered_indices[order_index]];
    OMNI_RETURN_VAL_IF(tensor.relative_offset_bytes % alignment_bytes != 0U,
                  SetError("GGUF tensor offset is not aligned: " + tensor.name));
    OMNI_RETURN_VAL_IF(model_.data_offset_bytes > UINT64_MAX - tensor.relative_offset_bytes,
                  SetError("GGUF tensor absolute offset overflow: " + tensor.name));
    tensor.data_offset_bytes = model_.data_offset_bytes + tensor.relative_offset_bytes;
    OMNI_RETURN_VAL_IF(tensor.data_offset_bytes > model_.file_size_bytes,
                  SetError("GGUF tensor offset exceeds file size: " + tensor.name));

    uint64_t next_offset_bytes = model_.file_size_bytes - model_.data_offset_bytes;
    if (order_index + 1U < ordered_indices.size()) {
      next_offset_bytes = model_.tensors[ordered_indices[order_index + 1U]].relative_offset_bytes;
    }
    OMNI_RETURN_VAL_IF(next_offset_bytes < tensor.relative_offset_bytes,
                  SetError("GGUF tensor offsets are invalid: " + tensor.name));
    tensor.storage_span_bytes = next_offset_bytes - tensor.relative_offset_bytes;

    GGUFTensorTypeInfo type_info;
    if (GetTensorTypeInfo(tensor.type_id, &type_info) && type_info.block_element_count != 0U &&
        type_info.block_size_bytes != 0U) {
      OMNI_RETURN_VAL_IF(tensor.dimensions.front() % type_info.block_element_count != 0U,
                    SetError("GGUF tensor row is incompatible with its data type: " + tensor.name));
      const uint64_t block_count = tensor.element_count / type_info.block_element_count;
      OMNI_RETURN_VAL_IF(!Multiply(block_count, type_info.block_size_bytes, &tensor.storage_size_bytes),
                    SetError("GGUF tensor storage size overflow: " + tensor.name));
      tensor.has_known_storage_size = true;
      OMNI_RETURN_VAL_IF(tensor.storage_size_bytes > tensor.storage_span_bytes,
                    SetError("GGUF tensor storage exceeds its data span: " + tensor.name));
    } else {
      tensor.storage_size_bytes = tensor.storage_span_bytes;
      tensor.has_known_storage_size = false;
    }
  }
  return true;
}

bool GGUFSerializer::ReadString(std::string *const output) {
  OMNI_RETURN_VAL_IF(output == nullptr, false);
  uint64_t size_bytes = 0U;
  OMNI_RETURN_VAL_IF(!reader_->Read(&size_bytes), false);
  OMNI_RETURN_VAL_IF(size_bytes > reader_->remaining_bytes() ||
                    size_bytes > static_cast<uint64_t>(SIZE_MAX),
                false);
  output->resize(static_cast<size_t>(size_bytes));
  if (size_bytes == 0U) {
    return true;
  }
  return reader_->ReadBytes(output->data(), output->size());
}

bool GGUFSerializer::ReadTensorBytes(const std::string &name, const uint64_t offset_bytes,
                                     const size_t size_bytes,
                                     std::vector<uint8_t> *const data) const {
  OMNI_RETURN_VAL_IF(data == nullptr || reader_ == nullptr || !is_parsed_, false);
  const GGUFTensorInfo *const tensor = FindTensor(name);
  OMNI_RETURN_VAL_IF(tensor == nullptr || offset_bytes > tensor->storage_size_bytes ||
                    size_bytes > tensor->storage_size_bytes - offset_bytes,
                false);
  OMNI_RETURN_VAL_IF(tensor->data_offset_bytes > UINT64_MAX - offset_bytes, false);
  data->resize(size_bytes);
  if (size_bytes == 0U) {
    return true;
  }
  return reader_->ReadAt(tensor->data_offset_bytes + offset_bytes, size_bytes, data->data());
}

bool GGUFSerializer::IsGGUFFile(const std::string &file_path) {
  Reader reader(file_path);
  OMNI_RETURN_VAL_IF(!reader.Open(), false);
  std::array<char, kGGUFMagic.size()> magic = {};
  OMNI_RETURN_VAL_IF(!reader.ReadBytes(magic.data(), magic.size()), false);
  return std::equal(magic.begin(), magic.end(), kGGUFMagic.begin());
}

bool GGUFSerializer::GetTensorTypeInfo(const uint32_t type_id,
                                       GGUFTensorTypeInfo *const type_info) {
  OMNI_RETURN_VAL_IF(type_info == nullptr, false);
  switch (static_cast<GGUFTensorType>(type_id)) {
  case GGUFTensorType::FLOAT32:
    return SetTensorTypeInfo(type_id, "float32", 1U, 4U, false, false, type_info);
  case GGUFTensorType::FLOAT16:
    return SetTensorTypeInfo(type_id, "float16", 1U, 2U, false, false, type_info);
  case GGUFTensorType::Q4_0:
    return SetTensorTypeInfo(type_id, "q4_0", 32U, 18U, true, false, type_info);
  case GGUFTensorType::Q4_1:
    return SetTensorTypeInfo(type_id, "q4_1", 32U, 20U, true, false, type_info);
  case GGUFTensorType::Q5_0:
    return SetTensorTypeInfo(type_id, "q5_0", 32U, 22U, true, false, type_info);
  case GGUFTensorType::Q5_1:
    return SetTensorTypeInfo(type_id, "q5_1", 32U, 24U, true, false, type_info);
  case GGUFTensorType::Q8_0:
    return SetTensorTypeInfo(type_id, "q8_0", 32U, 34U, true, false, type_info);
  case GGUFTensorType::Q8_1:
    return SetTensorTypeInfo(type_id, "q8_1", 32U, 40U, true, false, type_info);
  case GGUFTensorType::Q2_K:
    return SetTensorTypeInfo(type_id, "q2_k", 256U, 84U, true, false, type_info);
  case GGUFTensorType::Q3_K:
    return SetTensorTypeInfo(type_id, "q3_k", 256U, 110U, true, false, type_info);
  case GGUFTensorType::Q4_K:
    return SetTensorTypeInfo(type_id, "q4_k", 256U, 144U, true, false, type_info);
  case GGUFTensorType::Q5_K:
    return SetTensorTypeInfo(type_id, "q5_k", 256U, 176U, true, false, type_info);
  case GGUFTensorType::Q6_K:
    return SetTensorTypeInfo(type_id, "q6_k", 256U, 210U, true, false, type_info);
  case GGUFTensorType::Q8_K:
    return SetTensorTypeInfo(type_id, "q8_k", 256U, 292U, true, false, type_info);
  case GGUFTensorType::IQ2_XXS:
    return SetTensorTypeInfo(type_id, "iq2_xxs", 256U, 66U, true, false, type_info);
  case GGUFTensorType::IQ2_XS:
    return SetTensorTypeInfo(type_id, "iq2_xs", 256U, 74U, true, false, type_info);
  case GGUFTensorType::IQ3_XXS:
    return SetTensorTypeInfo(type_id, "iq3_xxs", 256U, 98U, true, false, type_info);
  case GGUFTensorType::IQ1_S:
    return SetTensorTypeInfo(type_id, "iq1_s", 256U, 50U, true, false, type_info);
  case GGUFTensorType::IQ4_NL:
    return SetTensorTypeInfo(type_id, "iq4_nl", 32U, 18U, true, false, type_info);
  case GGUFTensorType::IQ3_S:
    return SetTensorTypeInfo(type_id, "iq3_s", 256U, 110U, true, false, type_info);
  case GGUFTensorType::IQ2_S:
    return SetTensorTypeInfo(type_id, "iq2_s", 256U, 82U, true, false, type_info);
  case GGUFTensorType::IQ4_XS:
    return SetTensorTypeInfo(type_id, "iq4_xs", 256U, 136U, true, false, type_info);
  case GGUFTensorType::INT8:
    return SetTensorTypeInfo(type_id, "int8", 1U, 1U, false, false, type_info);
  case GGUFTensorType::INT16:
    return SetTensorTypeInfo(type_id, "int16", 1U, 2U, false, false, type_info);
  case GGUFTensorType::INT32:
    return SetTensorTypeInfo(type_id, "int32", 1U, 4U, false, false, type_info);
  case GGUFTensorType::INT64:
    return SetTensorTypeInfo(type_id, "int64", 1U, 8U, false, false, type_info);
  case GGUFTensorType::FLOAT64:
    return SetTensorTypeInfo(type_id, "float64", 1U, 8U, false, false, type_info);
  case GGUFTensorType::IQ1_M:
    return SetTensorTypeInfo(type_id, "iq1_m", 256U, 56U, true, false, type_info);
  case GGUFTensorType::BFLOAT16:
    return SetTensorTypeInfo(type_id, "bfloat16", 1U, 2U, false, false, type_info);
  case GGUFTensorType::TQ1_0:
    return SetTensorTypeInfo(type_id, "tq1_0", 256U, 54U, true, false, type_info);
  case GGUFTensorType::TQ2_0:
    return SetTensorTypeInfo(type_id, "tq2_0", 256U, 66U, true, false, type_info);
  case GGUFTensorType::MXFP4:
    return SetTensorTypeInfo(type_id, "mxfp4", 32U, 17U, true, false, type_info);
  case GGUFTensorType::NVFP4:
    return SetTensorTypeInfo(type_id, "nvfp4", 64U, sizeof(GGUFNVFP4Block), true, false, type_info);
  case GGUFTensorType::Q1_0:
    return SetTensorTypeInfo(type_id, "q1_0", 0U, 0U, true, false, type_info);
  case GGUFTensorType::Q2_0:
    return SetTensorTypeInfo(type_id, "q2_0", 0U, 0U, true, false, type_info);
  case GGUFTensorType::FP8_E4M3:
    return SetTensorTypeInfo(type_id, "fp8_e4m3", 1U, 1U, false, true, type_info);
  case GGUFTensorType::FP8_E5M2:
    return SetTensorTypeInfo(type_id, "fp8_e5m2", 1U, 1U, false, true, type_info);
  default:
    return false;
  }
}

bool GGUFSerializer::IsParsed() const {
  return is_parsed_;
}

bool GGUFSerializer::HasMetadata(const std::string &key) const {
  return metadata_index_.find(key) != metadata_index_.end();
}

bool GGUFSerializer::HasTensor(const std::string &name) const {
  return tensor_index_.find(name) != tensor_index_.end();
}

const GGUFModelInfo &GGUFSerializer::model() const {
  return model_;
}

const std::string &GGUFSerializer::error_message() const {
  return error_message_;
}

const GGUFMetadata *GGUFSerializer::FindMetadata(const std::string &key) const {
  const auto iterator = metadata_index_.find(key);
  OMNI_RETURN_VAL_IF(iterator == metadata_index_.end(), nullptr);
  return &model_.metadata[iterator->second];
}

const GGUFTensorInfo *GGUFSerializer::FindTensor(const std::string &name) const {
  const auto iterator = tensor_index_.find(name);
  OMNI_RETURN_VAL_IF(iterator == tensor_index_.end(), nullptr);
  return &model_.tensors[iterator->second];
}

bool GGUFSerializer::SetError(const std::string &message) {
  error_message_ = message;
  LOG_ERROR(message);
  return false;
}

bool GGUFSerializer::Reset() {
  reader_.reset();
  model_ = {};
  metadata_index_.clear();
  tensor_index_.clear();
  error_message_.clear();
  is_parsed_ = false;
  return true;
}

} // namespace util
} // namespace omni_runtime

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
