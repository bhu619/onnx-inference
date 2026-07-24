#include "qwen_image_processor.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kPatchSize = 16;
constexpr int kTemporalPatchSize = 2;
constexpr int kMergeSize = 2;
constexpr int kResizeFactor = kPatchSize * kMergeSize;
constexpr int64_t kMinPixels = 65536;
constexpr int64_t kMaxPixels = 16777216;

struct RgbImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

RgbImage LoadImage(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("image file not found: " + path.string());
  }

  int width = 0;
  int height = 0;
  stbi_uc* pixels = stbi_load(path.string().c_str(), &width, &height, nullptr, 3);
  if (pixels == nullptr) {
    const char* reason = stbi_failure_reason();
    throw std::runtime_error("cannot decode image: " + path.string() +
                             (reason == nullptr ? "" : ": " + std::string(reason)));
  }

  struct StbImageGuard {
    stbi_uc* value;
    ~StbImageGuard() { stbi_image_free(value); }
  } guard{pixels};
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("invalid image dimensions: " + path.string());
  }

  RgbImage image;
  image.width = width;
  image.height = height;
  image.pixels.assign(pixels, pixels + static_cast<size_t>(width) * height * 3);
  return image;
}

int RoundToMultiple(double value, int factor) {
  return std::max(factor, static_cast<int>(std::round(value / factor)) * factor);
}

int FloorToMultiple(double value, int factor) {
  return std::max(factor, static_cast<int>(std::floor(value / factor)) * factor);
}

int CeilToMultiple(double value, int factor) {
  return std::max(factor, static_cast<int>(std::ceil(value / factor)) * factor);
}

std::pair<int, int> SmartResize(int height, int width) {
  const double ratio =
      static_cast<double>(std::max(height, width)) / std::min(height, width);
  if (ratio > 200.0) {
    throw std::runtime_error("image aspect ratio must not exceed 200");
  }

  int resized_h = RoundToMultiple(height, kResizeFactor);
  int resized_w = RoundToMultiple(width, kResizeFactor);
  const int64_t pixels = static_cast<int64_t>(resized_h) * resized_w;
  if (pixels > kMaxPixels) {
    const double scale = std::sqrt(
        static_cast<double>(height) * width / static_cast<double>(kMaxPixels));
    resized_h = FloorToMultiple(height / scale, kResizeFactor);
    resized_w = FloorToMultiple(width / scale, kResizeFactor);
  } else if (pixels < kMinPixels) {
    const double scale = std::sqrt(
        static_cast<double>(kMinPixels) / (static_cast<double>(height) * width));
    resized_h = CeilToMultiple(height * scale, kResizeFactor);
    resized_w = CeilToMultiple(width * scale, kResizeFactor);
  }
  return {resized_h, resized_w};
}

double CubicWeight(double value) {
  constexpr double a = -0.5;
  value = std::abs(value);
  if (value <= 1.0) {
    return (a + 2.0) * value * value * value -
           (a + 3.0) * value * value + 1.0;
  }
  if (value < 2.0) {
    return a * value * value * value - 5.0 * a * value * value +
           8.0 * a * value - 4.0 * a;
  }
  return 0.0;
}

std::vector<float> ResizeAndNormalize(const RgbImage& image, int output_h,
                                      int output_w) {
  std::vector<float> output(
      static_cast<size_t>(output_h) * output_w * 3);
  const double scale_y = static_cast<double>(image.height) / output_h;
  const double scale_x = static_cast<double>(image.width) / output_w;
  for (int y = 0; y < output_h; ++y) {
    const double source_y = (y + 0.5) * scale_y - 0.5;
    const int base_y = static_cast<int>(std::floor(source_y));
    for (int x = 0; x < output_w; ++x) {
      const double source_x = (x + 0.5) * scale_x - 0.5;
      const int base_x = static_cast<int>(std::floor(source_x));
      for (int c = 0; c < 3; ++c) {
        double value = 0.0;
        double weight_sum = 0.0;
        for (int ky = -1; ky <= 2; ++ky) {
          const int sy = std::clamp(base_y + ky, 0, image.height - 1);
          const double wy = CubicWeight(source_y - (base_y + ky));
          for (int kx = -1; kx <= 2; ++kx) {
            const int sx = std::clamp(base_x + kx, 0, image.width - 1);
            const double weight =
                wy * CubicWeight(source_x - (base_x + kx));
            value += weight *
                     image.pixels[
                         (static_cast<size_t>(sy) * image.width + sx) * 3 + c];
            weight_sum += weight;
          }
        }
        value = weight_sum == 0.0 ? 0.0 : value / weight_sum;
        value = std::clamp(value, 0.0, 255.0);
        output[(static_cast<size_t>(y) * output_w + x) * 3 + c] =
            static_cast<float>(value / 127.5 - 1.0);
      }
    }
  }
  return output;
}

}  // namespace

QwenProcessedImage ProcessQwenImage(const std::filesystem::path& path) {
  const RgbImage image = LoadImage(path);
  const auto [resized_h, resized_w] = SmartResize(image.height, image.width);
  const std::vector<float> resized =
      ResizeAndNormalize(image, resized_h, resized_w);

  QwenProcessedImage processed;
  processed.grid_h = resized_h / kPatchSize;
  processed.grid_w = resized_w / kPatchSize;
  processed.pixel_values.reserve(
      static_cast<size_t>(processed.NumPatches()) * 3 * kTemporalPatchSize *
      kPatchSize * kPatchSize);

  const int block_h_count = processed.grid_h / kMergeSize;
  const int block_w_count = processed.grid_w / kMergeSize;
  for (int block_h = 0; block_h < block_h_count; ++block_h) {
    for (int block_w = 0; block_w < block_w_count; ++block_w) {
      for (int merge_h = 0; merge_h < kMergeSize; ++merge_h) {
        for (int merge_w = 0; merge_w < kMergeSize; ++merge_w) {
          for (int channel = 0; channel < 3; ++channel) {
            for (int temporal = 0; temporal < kTemporalPatchSize; ++temporal) {
              static_cast<void>(temporal);
              for (int patch_h = 0; patch_h < kPatchSize; ++patch_h) {
                const int y =
                    (block_h * kMergeSize + merge_h) * kPatchSize + patch_h;
                for (int patch_w = 0; patch_w < kPatchSize; ++patch_w) {
                  const int x =
                      (block_w * kMergeSize + merge_w) * kPatchSize + patch_w;
                  processed.pixel_values.push_back(
                      resized[(static_cast<size_t>(y) * resized_w + x) * 3 +
                              channel]);
                }
              }
            }
          }
        }
      }
    }
  }
  return processed;
}
