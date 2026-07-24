#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

struct QwenProcessedImage {
  std::vector<float> pixel_values;
  int64_t grid_t = 1;
  int64_t grid_h = 0;
  int64_t grid_w = 0;

  int64_t NumPatches() const {
    return grid_t * grid_h * grid_w;
  }

  int64_t NumFeatures() const {
    return grid_t * (grid_h / 2) * (grid_w / 2);
  }
};

// Loads an image through stb_image, resizes and normalizes
// it, then packs Qwen3.5 vision patches in the layout expected by the ONNX
// vision encoder.
QwenProcessedImage ProcessQwenImage(const std::filesystem::path& path);
