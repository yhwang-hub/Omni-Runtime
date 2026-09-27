#include "Omni-Runtime/inference/common/image_preprocessor/image_preprocessor.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace omni_runtime {
namespace inference {
namespace {

class ImagePreprocessorTest : public ::testing::Test {
protected:
  void SetUp() override {
    path_ = std::filesystem::path(::testing::TempDir()) /
            ("omni_image_" +
             std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove(path_, error);
    EXPECT_FALSE(error) << error.message();
  }

  bool WriteImage(const std::vector<uint8_t> &bytes) {
    std::ofstream stream(path_, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    return stream.good();
  }

  std::filesystem::path path_;
};

TEST_F(ImagePreprocessorTest, LoadsRgbWithoutChangingChannelOrder) {
  ASSERT_TRUE(
      WriteImage({'P', '6', '\n', '2', ' ', '1', '\n', '2', '5', '5', '\n', 255, 0, 7, 1, 128, 0}));
  std::vector<uint8_t> pixels;
  int32_t width_pixels = 0;
  int32_t height_pixels = 0;
  ASSERT_TRUE(ImagePreprocessor::LoadImage(path_.string(), &pixels, &width_pixels, &height_pixels));
  EXPECT_EQ(width_pixels, 2);
  EXPECT_EQ(height_pixels, 1);
  EXPECT_EQ(pixels, (std::vector<uint8_t>{255, 0, 7, 1, 128, 0}));
}

TEST_F(ImagePreprocessorTest, ConvertsGrayscaleToRgb) {
  ASSERT_TRUE(WriteImage({'P', '5', '\n', '2', ' ', '1', '\n', '2', '5', '5', '\n', 0, 127}));
  std::vector<uint8_t> pixels;
  int32_t width_pixels = 0;
  int32_t height_pixels = 0;
  ASSERT_TRUE(ImagePreprocessor::LoadImage(path_.string(), &pixels, &width_pixels, &height_pixels));
  EXPECT_EQ(width_pixels, 2);
  EXPECT_EQ(height_pixels, 1);
  EXPECT_EQ(pixels, (std::vector<uint8_t>{0, 0, 0, 127, 127, 127}));
}

TEST_F(ImagePreprocessorTest, LoadsJpegUsingRgbStorage) {
  const char *const test_srcdir = std::getenv("TEST_SRCDIR");
  const std::filesystem::path path =
      test_srcdir == nullptr
          ? std::filesystem::path("test/inference/demo.jpeg")
          : std::filesystem::path(test_srcdir) / "_main/test/inference/demo.jpeg";
  std::vector<uint8_t> pixels;
  int32_t width_pixels = 0;
  int32_t height_pixels = 0;
  for (const bool use_stb_decoder : {false, true}) {
    ASSERT_TRUE(ImagePreprocessor::LoadImage(path.string(), use_stb_decoder, &pixels, &width_pixels,
                                             &height_pixels));
    EXPECT_EQ(width_pixels, 2048);
    EXPECT_EQ(height_pixels, 1365);
    ASSERT_EQ(pixels.size(), 2048U * 1365U * 3U);
    EXPECT_EQ(pixels[0], 181U);
    EXPECT_EQ(pixels[1], 197U);
    EXPECT_EQ(pixels[2], 194U);
  }
}

TEST_F(ImagePreprocessorTest, RejectsMissingAndCorruptImagesWithoutChangingOutputs) {
  std::vector<uint8_t> pixels = {42};
  int32_t width_pixels = 7;
  int32_t height_pixels = 9;
  EXPECT_FALSE(
      ImagePreprocessor::LoadImage(path_.string(), &pixels, &width_pixels, &height_pixels));
  ASSERT_TRUE(WriteImage({0, 1, 2, 3}));
  EXPECT_FALSE(
      ImagePreprocessor::LoadImage(path_.string(), &pixels, &width_pixels, &height_pixels));
  ASSERT_TRUE(WriteImage({0xFF, 0xD8, 0xFF, 0xD9}));
  EXPECT_FALSE(
      ImagePreprocessor::LoadImage(path_.string(), &pixels, &width_pixels, &height_pixels));
  EXPECT_EQ(pixels, (std::vector<uint8_t>{42}));
  EXPECT_EQ(width_pixels, 7);
  EXPECT_EQ(height_pixels, 9);
}

TEST_F(ImagePreprocessorTest, RejectsNullOutputs) {
  std::vector<uint8_t> pixels;
  int32_t width_pixels = 0;
  int32_t height_pixels = 0;
  EXPECT_FALSE(ImagePreprocessor::LoadImage("", nullptr, &width_pixels, &height_pixels));
  EXPECT_FALSE(ImagePreprocessor::LoadImage("", &pixels, nullptr, &height_pixels));
  EXPECT_FALSE(ImagePreprocessor::LoadImage("", &pixels, &width_pixels, nullptr));
  EXPECT_FALSE(ImagePreprocessor::MakeGrid(32, 32, 16, 2, nullptr));
}

TEST(ImagePatchGridTest, PreservesQwen3VlReferenceDimensions) {
  ImagePatchGrid grid;
  ASSERT_TRUE(ImagePreprocessor::MakeGrid(2048, 1365, 16, 2, &grid));
  EXPECT_EQ(grid.width_patches, 128);
  EXPECT_EQ(grid.height_patches, 86);
  EXPECT_EQ(grid.token_count, 11008);
  EXPECT_EQ(grid.merged_token_count, 2752);
}

TEST(ImagePatchGridTest, AlignsTinyAndAlreadyAlignedImages) {
  ImagePatchGrid grid;
  ASSERT_TRUE(ImagePreprocessor::MakeGrid(1, 1, 14, 2, &grid));
  EXPECT_EQ(grid.width_patches, 2);
  EXPECT_EQ(grid.height_patches, 2);
  EXPECT_EQ(grid.merged_token_count, 1);
  ASSERT_TRUE(ImagePreprocessor::MakeGrid(56, 28, 14, 2, &grid));
  EXPECT_EQ(grid.width_patches, 4);
  EXPECT_EQ(grid.height_patches, 2);
  EXPECT_EQ(grid.token_count, 8);
  EXPECT_EQ(grid.merged_token_count, 2);
}

TEST(ImagePatchGridTest, RejectsInvalidAndOverflowingDimensionsWithoutChangingOutputs) {
  ImagePatchGrid grid;
  grid.token_count = 42;
  constexpr int32_t kMax = std::numeric_limits<int32_t>::max();
  EXPECT_FALSE(ImagePreprocessor::MakeGrid(0, 32, 16, 2, &grid));
  EXPECT_FALSE(ImagePreprocessor::MakeGrid(32, -1, 16, 2, &grid));
  EXPECT_FALSE(ImagePreprocessor::MakeGrid(32, 32, 0, 2, &grid));
  EXPECT_FALSE(ImagePreprocessor::MakeGrid(32, 32, 16, -1, &grid));
  EXPECT_FALSE(ImagePreprocessor::MakeGrid(kMax, 32, 16, 2, &grid));
  EXPECT_FALSE(ImagePreprocessor::MakeGrid(32, 32, kMax, kMax, &grid));
  EXPECT_FALSE(ImagePreprocessor::MakeGrid(65536, 65536, 1, 1, &grid));
  EXPECT_EQ(grid.token_count, 42);
}

} // namespace
} // namespace inference
} // namespace omni_runtime
