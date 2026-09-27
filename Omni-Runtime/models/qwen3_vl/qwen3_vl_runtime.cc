#include "Omni-Runtime/models/qwen3_vl/qwen3_vl_runtime.h"

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <utility>

#include "Omni-Runtime/utils/utils.h"

namespace omni_runtime {
namespace inference {
namespace qwen3_vl {

namespace {

using util::GGUFSerializer;
using util::GGUFTensorInfo;

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
  const float signed_output = sign == 0U ? output : -output;
  return signed_output;
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

memory::DataType TensorType(const ModelWeightType compute_type) {
  switch (compute_type) {
  case ModelWeightType::kFloat32:
    return memory::DataType::kFloat32;
  case ModelWeightType::kFloat16:
    return memory::DataType::kFloat16;
  case ModelWeightType::kBFloat16:
    return memory::DataType::kBFloat16;
  case ModelWeightType::kInt8:
    return memory::DataType::kUInt8;
  case ModelWeightType::kAuto:
    break;
  }
  return memory::DataType::kFloat32;
}

template <typename T>
bool UploadTensor(Qwen3VLRuntime *const runtime, const std::string &name,
                  const std::vector<uint64_t> &dimensions, const std::vector<T> &values,
                  const ModelWeightType compute_type) {
  OMNI_RETURN_VAL_IF(runtime == nullptr || values.empty(), false);
  std::vector<std::size_t> shape;
  shape.reserve(dimensions.size());
  for (const uint64_t dimension : dimensions) {
    shape.emplace_back(static_cast<std::size_t>(dimension));
  }
  const memory::TensorPtr tensor = memory::Tensor::Create(
      runtime->backend, shape, TensorType(compute_type), backend::MemoryKind::kDevice);
  OMNI_RETURN_VAL_IF(
      tensor == nullptr || !tensor->CopyFromHost(values.data(), values.size() * sizeof(T)), false);
  runtime->gpu_memory_bytes += tensor->size_bytes();
  runtime->weights.emplace(name, std::move(tensor));
  return true;
}

bool LoadWeight(const GGUFSerializer &serializer, Qwen3VLRuntime *const runtime,
                const Qwen3VLWeightName &weight) {
  OMNI_RETURN_VAL_IF(runtime == nullptr, false);
  if (runtime->weights.find(weight.name) != runtime->weights.end()) {
    return true;
  }
  const GGUFTensorInfo *const tensor = serializer.FindTensor(weight.name);
  OMNI_RETURN_VAL_IF(tensor == nullptr, false);
  const uint32_t type_id = tensor->type_id;
  if (runtime->config->compute_type() == ModelWeightType::kFloat32) {
    if (type_id == static_cast<uint32_t>(util::GGUFTensorType::FLOAT32)) {
      std::vector<float> values;
      return serializer.ReadTensorData<float>(
                 weight.name, 0U, static_cast<size_t>(tensor->element_count), &values) &&
             UploadTensor(runtime, weight.name, weight.dimensions, values,
                          ModelWeightType::kFloat32);
    }
    if (type_id == static_cast<uint32_t>(util::GGUFTensorType::FLOAT16)) {
      std::vector<util::GGUFFloat16> values;
      OMNI_RETURN_VAL_IF(!serializer.ReadTensorData<util::GGUFFloat16>(
                             weight.name, 0U, static_cast<size_t>(tensor->element_count), &values),
                         false);
      std::vector<float> converted;
      converted.reserve(values.size());
      for (const util::GGUFFloat16 value : values) {
        converted.emplace_back(HalfToFloat(value.bits));
      }
      return UploadTensor(runtime, weight.name, weight.dimensions, converted,
                          ModelWeightType::kFloat32);
    }
  }

  if (runtime->config->compute_type() == ModelWeightType::kBFloat16) {
    if (type_id == static_cast<uint32_t>(util::GGUFTensorType::BFLOAT16)) {
      std::vector<util::GGUFBFloat16> values;
      OMNI_RETURN_VAL_IF(!serializer.ReadTensorData<util::GGUFBFloat16>(
                             weight.name, 0U, static_cast<size_t>(tensor->element_count), &values),
                         false);
      std::vector<uint16_t> converted;
      converted.reserve(values.size());
      for (const util::GGUFBFloat16 value : values) {
        converted.emplace_back(value.bits);
      }
      return UploadTensor(runtime, weight.name, weight.dimensions, converted,
                          ModelWeightType::kBFloat16);
    }
    if (type_id == static_cast<uint32_t>(util::GGUFTensorType::FLOAT32)) {
      std::vector<float> values;
      OMNI_RETURN_VAL_IF(!serializer.ReadTensorData<float>(
                             weight.name, 0U, static_cast<size_t>(tensor->element_count), &values),
                         false);
      std::vector<uint16_t> converted;
      converted.reserve(values.size());
      for (const float value : values) {
        converted.emplace_back(FloatToBFloat16(value));
      }
      return UploadTensor(runtime, weight.name, weight.dimensions, converted,
                          ModelWeightType::kBFloat16);
    }
    if (type_id == static_cast<uint32_t>(util::GGUFTensorType::FLOAT16)) {
      std::vector<util::GGUFFloat16> values;
      OMNI_RETURN_VAL_IF(!serializer.ReadTensorData<util::GGUFFloat16>(
                             weight.name, 0U, static_cast<size_t>(tensor->element_count), &values),
                         false);
      std::vector<uint16_t> converted;
      converted.reserve(values.size());
      for (const util::GGUFFloat16 value : values) {
        converted.emplace_back(FloatToBFloat16(HalfToFloat(value.bits)));
      }
      return UploadTensor(runtime, weight.name, weight.dimensions, converted,
                          ModelWeightType::kBFloat16);
    }
  }

  if (runtime->config->compute_type() == ModelWeightType::kFloat16 &&
      type_id == static_cast<uint32_t>(util::GGUFTensorType::FLOAT16)) {
    std::vector<util::GGUFFloat16> values;
    OMNI_RETURN_VAL_IF(!serializer.ReadTensorData<util::GGUFFloat16>(
                           weight.name, 0U, static_cast<size_t>(tensor->element_count), &values),
                       false);
    return UploadTensor(runtime, weight.name, weight.dimensions, values, ModelWeightType::kFloat16);
  }
  if (runtime->config->compute_type() == ModelWeightType::kFloat16 &&
      type_id == static_cast<uint32_t>(util::GGUFTensorType::FLOAT32)) {
    std::vector<float> values;
    OMNI_RETURN_VAL_IF(!serializer.ReadTensorData<float>(
                           weight.name, 0U, static_cast<size_t>(tensor->element_count), &values),
                       false);
    std::vector<uint16_t> converted;
    converted.reserve(values.size());
    for (const float value : values) {
      converted.emplace_back(FloatToHalf(value));
    }
    return UploadTensor(runtime, weight.name, weight.dimensions, converted,
                        ModelWeightType::kFloat16);
  }
  return false;
}

} // namespace

size_t RuntimeElementSize(const Qwen3VLRuntime &runtime) {
  switch (runtime.config->compute_type()) {
  case ModelWeightType::kFloat32:
    return sizeof(float);
  case ModelWeightType::kFloat16:
  case ModelWeightType::kBFloat16:
    return sizeof(uint16_t);
  case ModelWeightType::kInt8:
    return sizeof(uint8_t);
  case ModelWeightType::kAuto:
    break;
  }
  return sizeof(float);
}

bool InitializeRuntime(Qwen3VLRuntime *const runtime, std::string *const error_message) {
  OMNI_RETURN_VAL_IF(runtime == nullptr || error_message == nullptr, false);
  OMNI_RETURN_VAL_IF(runtime->config == nullptr ||
                         runtime->config->model_backend() != ModelBackend::kGpu,
                     (*error_message = "Qwen3-VL supports only the GPU backend", false));
  runtime->backend = backend::CreateComputeBackend(runtime->config->compute_backend());
  OMNI_RETURN_VAL_IF(runtime->backend == nullptr || !runtime->backend->IsAvailable() ||
                         !runtime->compute_timer.Initialize(runtime->backend),
                     (*error_message = "failed to initialize the compute backend or timer", false));
  return true;
}

bool LoadRuntimeWeights(Qwen3VLRuntime *const runtime, std::string *const error_message) {
  OMNI_RETURN_VAL_IF(runtime == nullptr || error_message == nullptr, false);
  OMNI_RETURN_VAL_IF(runtime->model_serializer == nullptr ||
                         runtime->vision_serializer == nullptr || runtime->registry == nullptr,
                     (*error_message = "Qwen3-VL weight inputs are invalid", false));
  if (runtime->has_weights_loaded) {
    return true;
  }
  const Qwen3VLWeightRegistry &registry = *runtime->registry;
  const bool is_model_loaded =
      LoadWeight(*runtime->model_serializer, runtime, registry.token_embedding()) &&
      LoadWeight(*runtime->model_serializer, runtime, registry.output_norm()) &&
      LoadWeight(*runtime->model_serializer, runtime, registry.output_weight());
  OMNI_RETURN_VAL_IF(!is_model_loaded,
                     (*error_message = "failed to load Qwen3-VL output weights", false));
  for (const Qwen3VLLayerWeightNames &layer : registry.layers()) {
    const bool is_layer_loaded =
        LoadWeight(*runtime->model_serializer, runtime, layer.attention_norm) &&
        LoadWeight(*runtime->model_serializer, runtime, layer.query_norm) &&
        LoadWeight(*runtime->model_serializer, runtime, layer.key_norm) &&
        LoadWeight(*runtime->model_serializer, runtime, layer.query) &&
        LoadWeight(*runtime->model_serializer, runtime, layer.key) &&
        LoadWeight(*runtime->model_serializer, runtime, layer.value) &&
        LoadWeight(*runtime->model_serializer, runtime, layer.attention_output) &&
        LoadWeight(*runtime->model_serializer, runtime, layer.feed_forward_norm) &&
        LoadWeight(*runtime->model_serializer, runtime, layer.feed_forward_gate) &&
        LoadWeight(*runtime->model_serializer, runtime, layer.feed_forward_up) &&
        LoadWeight(*runtime->model_serializer, runtime, layer.feed_forward_down);
    OMNI_RETURN_VAL_IF(!is_layer_loaded,
                       (*error_message = "failed to load Qwen3-VL layer weights", false));
  }
  for (const Qwen3VLWeightName &weight : registry.vision_weights()) {
    OMNI_RETURN_VAL_IF(
        !LoadWeight(*runtime->vision_serializer, runtime, weight),
        (*error_message = "failed to load Qwen3-VL vision weight: " + weight.name, false));
  }
  runtime->model_config = registry.model_config();
  runtime->vision_config = registry.vision_config();
  OMNI_RETURN_VAL_IF(!runtime->backend->InitializeCompute() ||
                         !AllocateKvCache(runtime->config->max_context_tokens(), runtime) ||
                         !AllocateWorkspaces(runtime),
                     (*error_message = "failed to initialize Qwen3-VL compute storage", false));
  runtime->gpu_memory_bytes += runtime->backend->compute_workspace_bytes();
  runtime->has_weights_loaded = true;
  return true;
}

bool AllocateKvCache(const int32_t capacity, Qwen3VLRuntime *const runtime) {
  OMNI_RETURN_VAL_IF(runtime == nullptr || capacity <= 0, false);
  const int32_t key_value_size =
      runtime->model_config.key_value_head_count * runtime->model_config.head_size;
  const uint64_t element_count = static_cast<uint64_t>(runtime->model_config.layer_count) *
                                 static_cast<uint64_t>(capacity) *
                                 static_cast<uint64_t>(key_value_size);
  const size_t bytes = static_cast<size_t>(element_count) * RuntimeElementSize(*runtime);
  runtime->key_cache =
      memory::Buffer::CreateOwnedBuffer(runtime->backend, bytes, backend::MemoryKind::kDevice);
  runtime->value_cache =
      memory::Buffer::CreateOwnedBuffer(runtime->backend, bytes, backend::MemoryKind::kDevice);
  runtime->cache_capacity = capacity;
  runtime->gpu_memory_bytes += bytes * 2U;
  return runtime->key_cache != nullptr && runtime->value_cache != nullptr;
}

bool ResetGenerationState(Qwen3VLRuntime *const runtime) {
  OMNI_RETURN_VAL_IF(runtime == nullptr, false);
  runtime->used_token_count = 0;
  runtime->next_rope_position = 0;
  return true;
}

bool AllocateWorkspaces(Qwen3VLRuntime *const runtime) {
  OMNI_RETURN_VAL_IF(runtime == nullptr || runtime->config == nullptr, false);
  const size_t count = runtime->config->max_context_tokens();
  const size_t element_size = RuntimeElementSize(*runtime);
  const Qwen3VLConfig &language = runtime->model_config;
  const qwen3::Qwen3VisionConfig &vision = runtime->vision_config;
  Qwen3VLLanguageWorkspaces &l = runtime->language_workspaces;
  Qwen3VLVisionWorkspaces &v = runtime->vision_workspaces;
  const size_t vision_count = count * vision.merge_size * vision.merge_size;
  const size_t hidden_count = vision_count * vision.hidden_size;
  const auto allocate = [runtime](const size_t bytes, memory::BufferPtr *const buffer) {
    *buffer =
        memory::Buffer::CreateOwnedBuffer(runtime->backend, bytes, backend::MemoryKind::kDevice);
    OMNI_RETURN_VAL_IF(*buffer == nullptr, false);
    runtime->gpu_memory_bytes += bytes;
    return true;
  };
  const std::pair<memory::BufferPtr *, size_t> buffers[] = {
      {&runtime->decode_cache_position, sizeof(int32_t)},
      {&l.tokens, count * sizeof(int32_t)},
      {&l.positions, count * 3U * sizeof(int32_t)},
      {&l.hidden, count * language.hidden_size * element_size},
      {&l.next_hidden, count * language.hidden_size * element_size},
      {&l.normalized, count * language.hidden_size * element_size},
      {&l.query, count * language.query_head_count * language.head_size * element_size},
      {&l.key, count * language.key_value_head_count * language.head_size * element_size},
      {&l.value, count * language.key_value_head_count * language.head_size * element_size},
      {&l.query_normed, count * language.query_head_count * language.head_size * element_size},
      {&l.key_normed, count * language.key_value_head_count * language.head_size * element_size},
      {&l.attention, count * language.query_head_count * language.head_size * element_size},
      {&l.projected, count * language.hidden_size * element_size},
      {&l.gate, count * language.feed_forward_size * element_size},
      {&l.up, count * language.feed_forward_size * element_size},
      {&l.logits, static_cast<size_t>(language.vocabulary_size) * element_size},
      {&l.sample, sizeof(int32_t)},
      {&l.decode_partials, ((count + 127U) / 128U) * language.query_head_count *
                               (language.head_size + 2U) * sizeof(float)},
      {&v.source, vision_count * vision.patch_value_count()},
      {&v.patches, vision_count * vision.patch_value_count() * element_size},
      {&v.patch_embedding, hidden_count * element_size},
      {&v.hidden, hidden_count * element_size},
      {&v.residual, hidden_count * element_size},
      {&v.normalized, hidden_count * element_size},
      {&v.qkv, hidden_count * 3U * element_size},
      {&v.query, hidden_count * element_size},
      {&v.key, hidden_count * element_size},
      {&v.value, hidden_count * element_size},
      {&v.attention, hidden_count * element_size},
      {&v.projected, hidden_count * element_size},
      {&v.feed_forward, vision_count *
                            std::max(vision.feed_forward_size, vision.merged_hidden_size()) *
                            element_size},
      {&v.embedding, count * vision.projection_size * element_size},
  };
  for (const auto &entry : buffers) {
    OMNI_RETURN_VAL_IF(!allocate(entry.second, entry.first), false);
  }
  v.deepstack.resize(vision.deepstack_layers.size());
  for (memory::BufferPtr &buffer : v.deepstack) {
    OMNI_RETURN_VAL_IF(!allocate(count * vision.projection_size * element_size, &buffer), false);
  }
  return true;
}

backend::ScalarType RuntimeScalarType(const Qwen3VLRuntime &runtime) {
  switch (runtime.config->compute_type()) {
  case ModelWeightType::kFloat16:
    return backend::ScalarType::kFloat16;
  case ModelWeightType::kBFloat16:
    return backend::ScalarType::kBFloat16;
  default:
    return backend::ScalarType::kFloat32;
  }
}

} // namespace qwen3_vl
} // namespace inference
} // namespace omni_runtime

namespace omni_runtime::inference::qwen3_vl {

backend::DeviceView Weight(const Qwen3VLRuntime &runtime, const std::string &name) {
  const auto iterator = runtime.weights.find(name);
  return iterator == runtime.weights.end() ? backend::DeviceView{}
                                           : iterator->second->buffer()->view();
}

} // namespace omni_runtime::inference::qwen3_vl
