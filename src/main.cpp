// Shared entry point for the inference examples. Each executable links
// exactly one backend: QWEN_BACKEND_GENAI selects the ONNX Runtime GenAI
// backend (qwen3_infer); otherwise the plain ONNX Runtime backend
// (qwen3_5_ort) is used.

#if defined(QWEN_BACKEND_GENAI)
#include "qwen3_genai.h"
#else
#include "qwen3_5_ort.h"
#endif

int main(int argc, char** argv) {
#if defined(QWEN_BACKEND_GENAI)
  return run_qwen3_genai(argc, argv);
#else
  return run_qwen3_5_ort(argc, argv);
#endif
}
