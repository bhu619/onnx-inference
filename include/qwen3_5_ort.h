#pragma once

// Qwen3.5-0.8B inference on the ONNX Runtime C++ API (see qwen3_5_ort.cpp).
// Runs the full pipeline: argument parsing, BPE tokenization, chat template,
// ONNX graph execution and sampling decode. Returns the process exit code.
int run_qwen3_5_ort(int argc, char** argv);
