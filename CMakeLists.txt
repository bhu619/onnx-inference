cmake_minimum_required(VERSION 3.18)
project(qwen_onnx_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/third-party/onnxruntime" CACHE PATH
    "Path to the ONNX Runtime source tree")
set(ONNXRUNTIME_BUILD "${ONNXRUNTIME_ROOT}/build/Linux/RelWithDebInfo" CACHE PATH
    "Path containing the locally built libonnxruntime.so")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNX Runtime C++ headers not found at ${ONNXRUNTIME_ROOT}. "
    "Run 'git submodule update --init --recursive' or set ONNXRUNTIME_ROOT.")
endif()
find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS "${ONNXRUNTIME_BUILD}" NO_DEFAULT_PATH REQUIRED)

add_executable(qwen3_5_ort src/main.cpp src/qwen3_5_ort.cpp src/qwen_tokenizer.cpp)
target_include_directories(qwen3_5_ort PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
target_link_libraries(qwen3_5_ort PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(qwen3_5_ort PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_BUILD}"
  INSTALL_RPATH "${ONNXRUNTIME_BUILD}")

add_executable(inspect_onnx tools/inspect_onnx.cpp)
target_include_directories(inspect_onnx PRIVATE
  "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
target_link_libraries(inspect_onnx PRIVATE "${ONNXRUNTIME_LIBRARY}")
set_target_properties(inspect_onnx PROPERTIES BUILD_RPATH "${ONNXRUNTIME_BUILD}")

option(BUILD_ORT_GENAI_EXAMPLE "Also build the older Qwen3 ORT GenAI example" OFF)
if(BUILD_ORT_GENAI_EXAMPLE)
  set(ORT_GENAI_ROOT "${CMAKE_SOURCE_DIR}/third-party/onnxruntime-genai" CACHE PATH
      "Path to the onnxruntime-genai source tree")
  set(ORT_GENAI_BUILD "${ORT_GENAI_ROOT}/build/Linux/Release" CACHE PATH
      "Path containing libonnxruntime-genai.so and libonnxruntime.so")

  foreach(file ort_genai.h ort_genai_c.h)
    if(NOT EXISTS "${ORT_GENAI_ROOT}/src/${file}")
      message(FATAL_ERROR
        "Missing ${ORT_GENAI_ROOT}/src/${file}. "
        "Run 'git submodule update --init --recursive' or set ORT_GENAI_ROOT.")
    endif()
  endforeach()
  foreach(lib libonnxruntime-genai.so libonnxruntime.so)
    if(NOT EXISTS "${ORT_GENAI_BUILD}/${lib}")
      message(FATAL_ERROR "Missing ${ORT_GENAI_BUILD}/${lib}; set ORT_GENAI_BUILD")
    endif()
  endforeach()

  add_executable(qwen3_infer src/main.cpp src/qwen3_genai.cpp)
  target_compile_definitions(qwen3_infer PRIVATE QWEN_BACKEND_GENAI)
  target_include_directories(qwen3_infer PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${ORT_GENAI_ROOT}/src")
  target_link_libraries(qwen3_infer PRIVATE
    "${ORT_GENAI_BUILD}/libonnxruntime-genai.so"
    "${ORT_GENAI_BUILD}/libonnxruntime.so")
  set_target_properties(qwen3_infer PROPERTIES
    BUILD_RPATH "${ORT_GENAI_BUILD}"
    INSTALL_RPATH "${ORT_GENAI_BUILD}")
endif()
