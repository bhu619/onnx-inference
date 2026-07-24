#pragma once

// Qwen3-0.6B inference on ONNX Runtime GenAI (see qwen3_genai.cpp).
// Runs the full pipeline: argument parsing, chat template, generation and
// streaming decode. Returns the process exit code.
int run_qwen3_genai(int argc, char** argv);
