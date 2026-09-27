#include "kernels/backend_test_utils.h"

namespace omni_runtime::kernels {
namespace {

template <typename T> class ImageBackendTest : public test::BackendParityFixture {};
TYPED_TEST_SUITE(ImageBackendTest, test::BackendTypes);

TYPED_TEST(ImageBackendTest, BicubicResizePatchLayoutAndRoundingMatchCuda) {
  constexpr int32_t kWidth = 32;
  constexpr int32_t kHeight = 24;
  constexpr int32_t kPatch = 4;
  for (const bool has_resize : {false, true}) {
    const int32_t width = has_resize ? 29 : kWidth;
    const int32_t height = has_resize ? 19 : kHeight;
    std::vector<uint8_t> pixels(width * height * 3);
    for (size_t index = 0U; index < pixels.size(); ++index) {
      pixels[index] = (index * 37U + 13U) % 256U;
    }
    std::array<std::vector<float>, 2U> outputs;
    for (size_t device = 0U; device < this->devices().size(); ++device) {
      test::BackendTensor<uint8_t> input;
      test::BackendTensor<TypeParam> output;
      ASSERT_TRUE(input.Initialize(this->devices()[device], pixels));
      ASSERT_TRUE(output.Initialize(this->devices()[device],
                                    std::vector<TypeParam>(kWidth * kHeight * 3U)));
      ASSERT_TRUE(this->devices()[device]->Compute(
          test::BackendType<TypeParam>(),
          backend::ImagePatchPreprocess{.pixel_count = kWidth * kHeight,
                                        .source = input.view(),
                                        .source_width_pixels = kWidth,
                                        .source_height_pixels = kHeight,
                                        .patch_size = kPatch,
                                        .destination = output.view(),
                                        .input_width_pixels = has_resize ? width : 0,
                                        .input_height_pixels = has_resize ? height : 0}));
      ASSERT_TRUE(output.ReadFloats(&outputs[device]));
    }
    test::ExpectBackendClose(outputs[0], outputs[1], {1.0e-7F, 0.0F});
    if (!has_resize) {
      std::vector<float> expected(kWidth * kHeight * 3U);
      for (int32_t y = 0; y < kHeight; ++y) {
        for (int32_t x = 0; x < kWidth; ++x) {
          const int32_t patch = (y / kPatch) * (kWidth / kPatch) + x / kPatch;
          for (int32_t channel = 0; channel < 3; ++channel) {
            const int32_t target = patch * 3 * kPatch * kPatch + channel * kPatch * kPatch +
                                   (y % kPatch) * kPatch + x % kPatch;
            expected[target] = test::BackendRound<TypeParam>(
                pixels[(y * kWidth + x) * 3 + channel] / 127.5F - 1.0F);
          }
        }
      }
      test::ExpectBackendClose(outputs[1], expected, {1.0e-7F, 0.0F});
    }
  }
}

} // namespace
} // namespace omni_runtime::kernels
