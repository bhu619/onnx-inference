#include <iostream>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

const char* TypeName(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
    default: return "other";
  }
}

void PrintValue(const std::string& kind, size_t index, const char* name, const Ort::TypeInfo& info) {
  auto tensor = info.GetTensorTypeAndShapeInfo();
  std::cout << kind << '[' << index << "] " << name << " : "
            << TypeName(tensor.GetElementType()) << " [";
  auto shape = tensor.GetShape();
  auto symbols = tensor.GetSymbolicDimensions();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) std::cout << ", ";
    if (shape[i] >= 0) std::cout << shape[i];
    else if (symbols[i]) std::cout << symbols[i];
    else std::cout << '?';
  }
  std::cout << "]\n";
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " MODEL.onnx\n";
    return 2;
  }
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "inspect-onnx");
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    Ort::Session session(env, argv[1], options);
    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      PrintValue("input", i, name.get(), session.GetInputTypeInfo(i));
    }
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
      auto name = session.GetOutputNameAllocated(i, allocator);
      PrintValue("output", i, name.get(), session.GetOutputTypeInfo(i));
    }
  } catch (const Ort::Exception& e) {
    std::cerr << "ORT error: " << e.what() << '\n';
    return 1;
  }
}
