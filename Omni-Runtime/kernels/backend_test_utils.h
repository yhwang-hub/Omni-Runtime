#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels::test {

using BackendTypes = ::testing::Types<float, Float16, BFloat16>;

template <typename T> T BackendValue(const float value) {
  if constexpr (std::is_same_v<T, Float16> || std::is_same_v<T, BFloat16>) {
    return LowPrecision<T>(value);
  } else {
    return static_cast<T>(value);
  }
}

template <typename T> float BackendFloat(const T value) {
  if constexpr (std::is_same_v<T, Float16> || std::is_same_v<T, BFloat16>) {
    return HostFloat(value);
  } else {
    return static_cast<float>(value);
  }
}

template <typename T> float BackendRound(const float value) {
  return BackendFloat(BackendValue<T>(value));
}

template <typename T> constexpr backend::ScalarType BackendType() {
  if constexpr (std::is_same_v<T, float>) {
    return backend::ScalarType::kFloat32;
  } else if constexpr (std::is_same_v<T, Float16>) {
    return backend::ScalarType::kFloat16;
  } else {
    static_assert(std::is_same_v<T, BFloat16>);
    return backend::ScalarType::kBFloat16;
  }
}

template <typename T> std::vector<T> BackendInput(const size_t count, const int32_t salt) {
  std::vector<T> values(count);
  for (size_t index = 0U; index < count; ++index) {
    const int32_t numerator = (static_cast<int32_t>(index % 97U) * 17 + salt * 13) % 61 - 30;
    values[index] = BackendValue<T>(static_cast<float>(numerator) / 32.0F);
  }
  return values;
}

template <typename T> std::vector<float> BackendFloats(const std::vector<T> &values) {
  std::vector<float> result(values.size());
  for (size_t index = 0U; index < values.size(); ++index) {
    result[index] = BackendFloat(values[index]);
  }
  return result;
}

template <typename T>
void BackendRmsReference(const std::vector<T> &input, const std::vector<T> &weight,
                         const int32_t rows, const int32_t hidden_size, const float epsilon,
                         std::vector<float> *const output) {
  output->resize(input.size());
  for (int32_t row = 0; row < rows; ++row) {
    double squares = 0.0;
    for (int32_t dimension = 0; dimension < hidden_size; ++dimension) {
      const float value = BackendFloat(input[row * hidden_size + dimension]);
      squares += static_cast<double>(value) * value;
    }
    const float inverse = 1.0F / std::sqrt(static_cast<float>(squares / hidden_size) + epsilon);
    for (int32_t dimension = 0; dimension < hidden_size; ++dimension) {
      const size_t index = row * hidden_size + dimension;
      const float normalized = BackendRound<T>(BackendFloat(input[index]) * inverse);
      (*output)[index] = BackendRound<T>(normalized * BackendFloat(weight[dimension]));
    }
  }
}

struct BackendTolerance final {
  float absolute;
  float relative;
};

template <typename T> constexpr BackendTolerance BackendMathTolerance() {
  if constexpr (std::is_same_v<T, float>) {
    return {2.0e-5F, 2.0e-5F};
  } else if constexpr (std::is_same_v<T, Float16>) {
    return {1.0e-3F, 1.0e-3F};
  } else {
    return {8.0e-3F, 8.0e-3F};
  }
}

void ExpectBackendClose(const std::vector<float> &actual, const std::vector<float> &expected,
                        const BackendTolerance tolerance);

// A Buffer view and a further DeviceView slice exercise both offset-composition paths.
template <typename T> class BackendTensor final {
public:
  bool Initialize(const std::shared_ptr<backend::Backend> &device, const std::vector<T> &values) {
    count_ = values.size();
    std::vector<T> padded(count_ + kPrefix + kSuffix, BackendValue<T>(97.0F));
    for (size_t index = 0U; index < count_; ++index) {
      padded[kPrefix + index] = values[index];
    }
    storage_ = memory::Buffer::CreateDeviceBufferFromVector(device, padded);
    if (storage_ == nullptr) {
      return false;
    }
    buffer_ = memory::Buffer::CreateDeviceBufferView(storage_, sizeof(T),
                                                     (padded.size() - 2U) * sizeof(T));
    return buffer_ != nullptr && view().IsValid();
  }

  backend::DeviceView view() const {
    return buffer_->view().Slice((kPrefix - 1U) * sizeof(T), count_ * sizeof(T));
  }

  bool Read(std::vector<T> *const values) const {
    if (values == nullptr) {
      return false;
    }
    std::vector<T> padded;
    if (!storage_->CopyDataToVector(&padded)) {
      return false;
    }
    for (size_t index = 0U; index < padded.size(); ++index) {
      if (index < kPrefix || index >= kPrefix + count_) {
        EXPECT_EQ(BackendFloat(padded[index]), 97.0F) << "guard element " << index;
      }
    }
    values->assign(padded.begin() + kPrefix, padded.begin() + kPrefix + count_);
    return true;
  }

  bool ReadFloats(std::vector<float> *const values) const {
    if (values == nullptr) {
      return false;
    }
    std::vector<T> typed;
    if (!Read(&typed)) {
      return false;
    }
    *values = BackendFloats(typed);
    return true;
  }

private:
  static constexpr size_t kPrefix = 3U;
  static constexpr size_t kSuffix = 5U;
  size_t count_ = 0U;
  memory::BufferPtr storage_;
  memory::BufferPtr buffer_;
};

class BackendParityFixture : public ::testing::Test {
protected:
  void SetUp() override;
  const std::array<std::shared_ptr<backend::Backend>, 2U> &devices() const;
};

} // namespace omni_runtime::kernels::test
