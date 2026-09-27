#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/kernel_sources.h"
#include "kernels/kernel_test_utils.h"

namespace omni_runtime::kernels {
namespace {

constexpr int32_t kRows = 8;
constexpr int32_t kInnerSize = 4096;
constexpr int32_t kColumns = 4096;

class MatmulKernelTest : public test::KernelTestFixture {};

TEST_F(MatmulKernelTest, ProjectsQwen3VlHiddenState) {
  std::vector<float> lhs(static_cast<std::size_t>(kRows) * kInnerSize);
  std::vector<float> rhs(static_cast<std::size_t>(kInnerSize) * kColumns);
  std::vector<float> expected(static_cast<std::size_t>(kRows) * kColumns);
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t inner = 0; inner < kInnerSize; ++inner) {
      lhs[static_cast<int64_t>(row) * kInnerSize + inner] =
          0.0001f * static_cast<float>(inner) + 0.01f * static_cast<float>(row);
    }
  }
  for (int32_t inner = 0; inner < kInnerSize; ++inner) {
    for (int32_t column = 0; column < kColumns; ++column) {
      rhs[static_cast<int64_t>(inner) * kColumns + column] =
          0.00001f * static_cast<float>(column) + 0.001f * static_cast<float>(inner);
    }
  }
  for (int32_t row = 0; row < kRows; ++row) {
    for (int32_t column = 0; column < kColumns; ++column) {
      float value = 0.0f;
      for (int32_t inner = 0; inner < kInnerSize; ++inner) {
        value += lhs[static_cast<int64_t>(row) * kInnerSize + inner] *
                 rhs[static_cast<int64_t>(inner) * kColumns + column];
      }
      expected[static_cast<int64_t>(row) * kColumns + column] = value;
    }
  }
  const auto lhs_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), lhs);
  const auto rhs_device = memory::Buffer::CreateDeviceBufferFromVector(backend(), rhs);
  const auto output_device = memory::Buffer::CreateDeviceBuffer<float>(backend(), expected.size());
  ASSERT_TRUE(Compile("matmul_float", test::MakeMatmulFloatSource()));
  ASSERT_TRUE(Launch(dim3((kColumns + 15) / 16, (kRows + 15) / 16, 1), dim3(16, 16, 1), 0,
                     lhs_device->device_address(), rhs_device->device_address(), kRows, kInnerSize,
                     kColumns, output_device->device_address()));
  ASSERT_TRUE(Sync());
  std::vector<float> actual(expected.size());
  ASSERT_TRUE(output_device->CopyDataToVector(&actual));
  EXPECT_TRUE(test::IsClose(actual, expected, 2.0e-3f));
}
} // namespace
} // namespace omni_runtime::kernels
