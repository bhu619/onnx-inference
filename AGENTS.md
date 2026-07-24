# AGENTS.md

面向 AI 编码代理的项目说明。人类读者请直接看 [README.md](README.md)。

## 项目概述

基于 ONNX Runtime 的 Qwen 系列模型 C++ 推理示例，不依赖 Python / Transformers，
直接通过 ONNX Runtime C++ API 完成分词、Chat Template、图执行与采样解码。

## 构建目标

| 目标 | 源文件 | 说明 |
| --- | --- | --- |
| `qwen3_5_ort` | `src/main.cpp` + `src/qwen3_5_ort.cpp` + `src/qwen_tokenizer.cpp` | 默认示例，Qwen3.5-0.8B，纯 ONNX Runtime C++ API |
| `inspect_onnx` | `tools/inspect_onnx.cpp` | ONNX 模型输入输出检查工具 |
| `qwen3_infer` | `src/main.cpp` + `src/qwen3_genai.cpp` | 可选示例，Qwen3-0.6B，依赖 ONNX Runtime GenAI（编译宏 `QWEN_BACKEND_GENAI`） |

两个推理示例相互独立：`qwen3_5_ort` 不链接 GenAI，`qwen3_infer` 不使用
`qwen_tokenizer`。

## 构建

```bash
./scripts/init_submodules.sh --all   # 初始化 third-party 子模块（含嵌套子模块）

./third-party/onnxruntime/build.sh \
  --config RelWithDebInfo \
  --build_shared_lib \
  --parallel \
  --skip_tests

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

- ONNX Runtime 和 ONNX Runtime GenAI 位于 `third-party/` Git 子模块，
  分别固定为 `v1.27.1` 和 `v0.14.0`。
- `ONNXRUNTIME_ROOT` / `ONNXRUNTIME_BUILD` 默认指向 ONNX Runtime 子模块及其
  `build/Linux/RelWithDebInfo` 构建目录；仍可通过 `-D` 覆盖。
- 构建 `qwen3_infer` 前先运行
  `./third-party/onnxruntime-genai/build.sh --config Release --parallel --skip_tests`，
  再向 CMake 追加 `-DBUILD_ORT_GENAI_EXAMPLE=ON`。其根目录和构建目录也可通过
  `ORT_GENAI_ROOT` / `ORT_GENAI_BUILD` 覆盖。
- CMake 会写入 RUNPATH，运行时无需设置 `LD_LIBRARY_PATH`。

## 验证

仓库没有自动化测试。改动后按以下方式验证：

1. `cmake --build build -j` 编译通过（需先按上文构建 ONNX Runtime 子模块）。
2. 涉及 `qwen3_5_ort` 或分词器的改动，用真实模型跑一次推理确认输出正常：

   ```bash
   ./build/qwen3_5_ort --prompt "用三句话介绍一下北京" --max-new-tokens 32 --threads 4
   ```

   默认模型路径 `/home/ubuntu/.cache/models/Qwen3.5-0.8B-ONNX-OPT`，否则加 `--model`。
3. 涉及 ONNX 图接口（输入输出名称、维度、state 数量）的改动，用
   `./build/inspect_onnx <model.onnx>` 核对。

## 代码约定

- C++17，2 空格缩进；源码注释采用英文，README 等文档用中文。
- `src/main.cpp` 是两个二进制的共享入口，通过编译宏 `QWEN_BACKEND_GENAI`
  分发到对应后端；推理逻辑分别位于 `src/qwen3_5_ort.cpp` 与
  `src/qwen3_genai.cpp`，公共接口见 `include/` 下的同名头文件。
- 改动尽量小：两个示例相互独立，不要为它们引入不必要的共享抽象；
  分词器改动只进 `src/qwen_tokenizer.cpp` 与 `include/qwen_tokenizer.h`。
- 修改构建选项、目录结构、命令行参数或功能特性时，同步更新 README.md。

## 注意事项

- `models/`、`build/` 已被 `.gitignore` 忽略；模型权重和编译产物不入库。
- `.onnx`（计算图）与同名 `.onnx_data`（权重）必须位于同一目录。
- `qwen3_infer` 默认直接使用已转换的
  `xiaoyao9184/Qwen3-0.6B-onnx-genai` CPU INT4 设备子目录，不运行转换脚本；
  传给 `--model` 的目录必须直接包含 `genai_config.json`。
- `src/qwen3_5_ort.cpp` 中的 EOS / vocab size 等常量与 Qwen3.5-0.8B 模型绑定，
  更换模型时需同步核对。
