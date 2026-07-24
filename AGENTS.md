# AGENTS.md

面向 AI 编码代理的项目说明。人类读者请阅读 [README.md](README.md)。

## 项目概述

基于 ONNX Runtime 的 Qwen 系列模型 C++ 推理示例。不依赖 Python / Transformers，
直接通过 ONNX Runtime C++ API 完成分词、Chat Template、图执行与采样解码。
两个示例均以交互式终端会话运行，支持多轮输入、流式着色输出与 Ctrl+C 中断。

## 仓库结构

```text
├── include/       # 公共头文件：推理接口、分词器、图片预处理、终端 UI
├── src/           # 共享入口 main.cpp 与各推理后端实现
├── tools/         # inspect_onnx：ONNX 模型输入输出检查工具
├── scripts/       # init_submodules.sh：third-party 子模块初始化脚本
├── third-party/   # Git 子模块：onnxruntime、onnxruntime-genai、stb
└── CMakeLists.txt
```

## 构建目标

| 目标 | 源文件 | 说明 |
| --- | --- | --- |
| `onnx-cli` | `src/main.cpp` `src/qwen3_5_ort.cpp` `src/qwen_image_processor.cpp` `src/qwen_tokenizer.cpp` | 默认示例：Qwen3.5-0.8B，纯 ONNX Runtime C++ API，支持图片输入 |
| `onnx-genai-cli` | `src/main.cpp` `src/qwen3_genai.cpp` | 可选示例：Qwen3-0.6B，依赖 ONNX Runtime GenAI（编译宏 `QWEN_BACKEND_GENAI`） |
| `inspect_onnx` | `tools/inspect_onnx.cpp` | ONNX 模型输入输出检查工具 |

两个推理示例相互独立：`onnx-cli` 不链接 GenAI，`onnx-genai-cli` 不使用分词器与
图片预处理模块；二者仅共享入口 `src/main.cpp` 和终端 UI `include/terminal_ui.h`。

## 依赖

依赖以 Git 子模块形式固定在 `third-party/`：

| 子模块 | 版本 | 用途 |
| --- | --- | --- |
| `third-party/onnxruntime` | `v1.27.1` | `onnx-cli` 与 `inspect_onnx` 的推理引擎 |
| `third-party/onnxruntime-genai` | `v0.14.0` | `onnx-genai-cli` 的生成接口与 Chat Template |
| `third-party/stb` | `31c1ad37456438565541f4919958214b6e762fb4` | `onnx-cli` 通过 `stb_image` 解码输入图片 |

初始化（含 ONNX Runtime 的嵌套子模块）：

```bash
./scripts/init_submodules.sh --all      # 全部初始化；-s <名称> 可选单个子模块，--dry-run 预览
```

## 构建

先构建依赖子模块，再编译示例程序：

```bash
# 1. ONNX Runtime（onnx-cli 与 inspect_onnx 依赖）
./third-party/onnxruntime/build.sh \
  --config RelWithDebInfo --build_shared_lib --parallel --skip_tests

# 2. ONNX Runtime GenAI（仅 onnx-genai-cli 需要）
./third-party/onnxruntime-genai/build.sh --config Release --parallel --skip_tests

# 3. 示例程序
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

CMake 选项：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `ONNXRUNTIME_ROOT` | `third-party/onnxruntime` | ONNX Runtime 源码目录 |
| `ONNXRUNTIME_BUILD` | `<ONNXRUNTIME_ROOT>/build/Linux/RelWithDebInfo` | 包含 `libonnxruntime.so` 的构建目录 |
| `BUILD_ORT_GENAI_EXAMPLE` | `OFF` | 设为 `ON` 才构建 `onnx-genai-cli` |
| `ORT_GENAI_ROOT` | `third-party/onnxruntime-genai` | GenAI 源码目录（仅可选示例） |
| `ORT_GENAI_BUILD` | `<ORT_GENAI_ROOT>/build/Linux/Release` | 包含 `libonnxruntime-genai.so` 的构建目录 |

CMake 会写入 RUNPATH，运行时无需设置 `LD_LIBRARY_PATH`。

## 验证

仓库没有自动化测试。改动后按以下方式验证：

1. `cmake --build build -j$(nproc)` 编译通过（需先完成上述依赖构建）。
2. 涉及 `onnx-cli`、分词器或终端交互的改动，用真实模型跑一次推理确认输出正常：

   ```bash
   ./build/onnx-cli --prompt "用三句话介绍一下北京" --max-new-tokens 32 --threads 4
   ```

   默认模型路径 `~/.cache/models/Qwen3.5-0.8B-ONNX-OPT`，否则加 `--model`。
3. 涉及图片预处理或视觉推理的改动，用真实图片验证：

   ```bash
   ./build/onnx-cli --image /path/to/image.jpg --prompt "描述这张图片" \
     --max-new-tokens 32 --threads 4
   ```

4. 涉及 ONNX 图接口（输入输出名称、维度、state 数量）的改动，用
   `./build/inspect_onnx <model.onnx>` 核对。

## 代码约定

- C++17，2 空格缩进；源码注释使用英文，README 等文档使用中文。
- `src/main.cpp` 是两个二进制的共享入口，通过编译宏 `QWEN_BACKEND_GENAI`
  分发到对应后端；后端公共接口为 `include/` 下的同名头文件。
- 改动保持最小，按职责归位：
  - 推理逻辑只进 `src/qwen3_5_ort.cpp` / `src/qwen3_genai.cpp`；
  - 分词器只进 `src/qwen_tokenizer.cpp` 与 `include/qwen_tokenizer.h`；
  - 图片预处理只进 `src/qwen_image_processor.cpp` 与 `include/qwen_image_processor.h`；
  - 与后端无关的交互终端逻辑只进 `include/terminal_ui.h`。
- 修改构建选项、目录结构、命令行参数或功能特性时，同步更新 README.md 与本文件。

## 注意事项

- `build/`、`models/`、`third-party/` 已被 `.gitignore` 忽略；模型权重与编译产物不入库。
- `.onnx`（计算图）与同名 `.onnx_data`（权重）必须位于同一目录。
- 两个示例的默认模型路径由 `terminal_ui::ModelCachePath` 解析到
  `$HOME/.cache/models/` 下的固定子目录；`HOME` 未设置时报错并提示使用 `--model`。
- `onnx-genai-cli` 默认直接使用已转换的
  `xiaoyao9184/Qwen3-0.6B-onnx-genai` CPU INT4 设备子目录，不运行转换脚本；
  传给 `--model` 的目录必须直接包含 `genai_config.json`。
- `src/qwen3_5_ort.cpp` 中的 EOS（248044）、im_end（248046）、vocab size（248320）
  等常量与 Qwen3.5-0.8B 模型绑定，更换模型时需同步核对。
