# onnx-inference

基于 [ONNX Runtime](https://github.com/microsoft/onnxruntime) 的 Qwen 系列模型 C++ 推理示例。不依赖 Python 与
Transformers，直接通过 ONNX Runtime C++ API 完成分词、Chat Template、
图执行与采样解码的完整推理流程。

## 1. 示例程序

| 程序 | 目标模型 | 推理后端 | 默认构建 |
| --- | --- | --- | --- |
| `qwen3_5_ort` | Qwen3.5-0.8B | ONNX Runtime（C++ API） | 是 |
| `qwen3_infer` | Qwen3-0.6B | ONNX Runtime GenAI | 否（需显式启用） |
| `inspect_onnx` | — | ONNX Runtime（模型结构检查工具） | 是 |

两个推理示例相互独立，可单独构建和使用。

---

## 2. 功能特性

`qwen3_5_ort` 直接调用本地编译的 ONNX Runtime C++ API，不链接
ONNX Runtime GenAI，也不依赖 `genai_config.json`：

- Qwen ByteLevel BPE 分词与流式解码
- Chat Template 拼接（支持思考 / 非思考模式）
- Embedding 与 Decoder 两张 ONNX 图的执行循环
- 18 组 recurrent state 与 6 组 KV Cache 的维护
- MRoPE 三维 `position_ids`
- Greedy、top-k、top-p、temperature 与 presence penalty 解码策略
- EOS 检测与推理性能统计

当前示例仅支持纯文本推理，不会调用 `vision_encoder_q4.onnx`。

---

## 3. 目录结构

```text
├── src/
│   ├── qwen3_5_ort.cpp         # Qwen3.5-0.8B 推理主程序（默认示例）
│   ├── qwen_tokenizer.{h,cpp} # Qwen BPE 分词器
│   └── main.cpp              # Qwen3-0.6B 推理程序（ORT GenAI 示例）
├── tools/
│   └── inspect_onnx.cpp      # ONNX 模型输入输出检查工具
├── scripts/
│   └── init_submodules.sh    # 初始化 third-party Git 子模块
├── third-party/
│   ├── onnxruntime/          # Git 子模块，固定为 v1.27.1
│   └── onnxruntime-genai/    # Git 子模块，固定为 v0.14.0
└── CMakeLists.txt
```

---

## 4. 环境依赖

- CMake >= 3.18，支持 C++17 的编译器
- Git
- （可选）`hf` 命令行工具（`pip install -U "huggingface_hub[cli]"`），
  仅下载模型时需要

推理依赖的 ONNX Runtime 与 ONNX Runtime GenAI 以 Git 子模块形式放在
`third-party/`，固定在以下版本：

| 子模块 | 版本 | 简介 |
| --- | --- | --- |
| [`third-party/onnxruntime`](https://github.com/microsoft/onnxruntime) | `v1.27.1` | 跨平台 ONNX 推理引擎；`qwen3_5_ort` 与 `inspect_onnx` 直接调用其 C++ API |
| [`third-party/onnxruntime-genai`](https://github.com/microsoft/onnxruntime-genai) | `v0.14.0` | 基于 ONNX Runtime 的生成式推理库；`qwen3_infer` 依赖其 Chat Template 与生成接口 |

---

## 5. 初始化依赖

克隆主仓库时加 `--recurse-submodules` 可一次拉取全部子模块：

```bash
git clone --recurse-submodules <本仓库地址>
cd onnx-inference
```

已通过普通 `git clone` 下载的仓库，可用以下两种方式初始化子模块。
二者都会把子模块检出到主仓库固定的提交，并递归初始化 ONNX Runtime
的嵌套子模块，不会擅自更新到远端分支的最新提交。

### 5.1 使用初始化脚本（推荐）

```bash
./scripts/init_submodules.sh
```

脚本会先同步 `.gitmodules` 中的 URL，再执行递归初始化。主仓库更新
子模块版本后，重新运行该脚本即可。

### 5.2 使用 Git 命令

与脚本等价的手动命令：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

初始化完成后，可用下面的命令核对当前固定的版本：

```bash
git submodule status
```

---

## 6. 快速开始：Qwen3.5-0.8B（`qwen3_5_ort`）

### 6.1 准备模型文件

模型托管在 Hugging Face：
[onnx-community/Qwen3.5-0.8B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT/tree/main)。
使用 `hf` 命令（由 `pip install -U "huggingface_hub[cli]"` 提供）下载到当前目录：

```bash
hf download onnx-community/Qwen3.5-0.8B-ONNX-OPT \
  --include "chat_template.jinja" \
  --include "config.json" \
  --include "generation_config.json" \
  --include "tokenizer.json" \
  --include "tokenizer_config.json" \
  --include "onnx/embed_tokens_q4.onnx*" \
  --include "onnx/decoder_model_merged_q4.onnx*" \
  --include "preprocessor_config.json" \
  --include "processor_config.json" \
  --include "onnx/vision_encoder_q4.onnx*" \
  --local-dir .
```

推理实际需要的文件如下（其余为可选的配置与多模态文件，当前示例不会调用
`vision_encoder_q4.onnx`）：

```text
Qwen3.5-0.8B-ONNX-OPT/
├── tokenizer.json
└── onnx/
    ├── embed_tokens_q4.onnx
    ├── embed_tokens_q4.onnx_data
    ├── decoder_model_merged_q4.onnx
    └── decoder_model_merged_q4.onnx_data
```

`.onnx` 保存计算图，`.onnx_data` 保存模型权重，二者必须位于同一目录。

程序默认从 `/home/ubuntu/.cache/models/Qwen3.5-0.8B-ONNX-OPT` 加载模型；
模型位于其他路径时，通过 `--model` 指定。

### 6.2 编译

先编译 ONNX Runtime 子模块（仅需一次）：

```bash
./third-party/onnxruntime/build.sh \
  --config RelWithDebInfo \
  --build_shared_lib \
  --parallel \
  --skip_tests
```

再编译示例程序：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j
```

产物为 `build/qwen3_5_ort`。CMake 会写入 ONNX Runtime 的 RUNPATH，
通常无需再设置 `LD_LIBRARY_PATH`。默认使用
`third-party/onnxruntime/build/Linux/RelWithDebInfo`；如需使用外部构建，
仍可通过 `-DONNXRUNTIME_ROOT=... -DONNXRUNTIME_BUILD=...` 覆盖。

### 6.3 运行

基本推理（Qwen3.5-0.8B 默认使用非思考模式）：

```bash
./build/qwen3_5_ort \
  --prompt "用三句话介绍一下北京" \
  --max-new-tokens 128 \
  --threads 4
```

启用思考模式：

```bash
./build/qwen3_5_ort \
  --prompt "计算 17 * 29，并解释计算过程" \
  --think \
  --max-new-tokens 256 \
  --threads 4
```

启用采样（默认为 Greedy 解码，`--sample` 开启 top-k/top-p 采样）：

```bash
./build/qwen3_5_ort \
  --prompt "写一个关于月亮的短故事" \
  --sample \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --presence-penalty 1.5 \
  --max-new-tokens 128 \
  --threads 4
```

### 6.4 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--model DIR` | 模型根目录 |
| `--prompt TEXT` | 用户提示词；省略时从标准输入读取一行 |
| `--system TEXT` | System 消息 |
| `--max-new-tokens N` | 最大生成 Token 数 |
| `--threads N` | ONNX Runtime CPU intra-op 线程数 |
| `--sample` | 启用 top-k/top-p 采样 |
| `--temperature N` | 采样温度 |
| `--top-k N` | top-k |
| `--top-p N` | top-p |
| `--presence-penalty N` | 对已经生成过的 Token 施加惩罚 |
| `--seed N` | 随机种子 |
| `--think` | 启用 Qwen3.5 思考模式 |
| `-h`, `--help` | 显示帮助 |

---

## 7. 可选示例：Qwen3-0.6B（`qwen3_infer`）

`qwen3_infer` 基于 ONNX Runtime GenAI，与 `qwen3_5_ort` 相互独立，
默认不参与构建。

### 7.1 下载模型

直接使用 Hugging Face 上的
[`xiaoyao9184/Qwen3-0.6B-onnx-genai`](https://huggingface.co/xiaoyao9184/Qwen3-0.6B-onnx-genai)。
该仓库已经使用 ONNX Runtime GenAI Builder 完成转换，设备子目录中包含
`genai_config.json`、ONNX 图、权重、Tokenizer 和 Chat Template，不需要自行转换：

```bash
hf download xiaoyao9184/Qwen3-0.6B-onnx-genai \
  --include "cpu_and_mobile/cpu-int4-rtn-block-32/*" \
  --local-dir /home/ubuntu/.cache/models/Qwen3-0.6B-onnx-genai
```

下载后目录如下：

```text
/home/ubuntu/.cache/models/Qwen3-0.6B-onnx-genai/
└── cpu_and_mobile/
    └── cpu-int4-rtn-block-32/
        ├── chat_template.jinja
        ├── genai_config.json
        ├── model.onnx
        ├── model.onnx.data
        ├── tokenizer.json
        └── tokenizer_config.json
```

`qwen3_infer` 默认使用上述 CPU INT4 子目录。其他设备版本必须通过
`--model` 指向相应的、直接包含 `genai_config.json` 的子目录。

### 7.2 编译

先编译 ONNX Runtime GenAI 子模块（仅需一次）。该项目会按其固定依赖
构建配套的 ONNX Runtime：

```bash
./third-party/onnxruntime-genai/build.sh \
  --config Release \
  --parallel \
  --skip_tests
```

再通过 `BUILD_ORT_GENAI_EXAMPLE` 显式启用并编译示例程序：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_ORT_GENAI_EXAMPLE=ON

cmake --build build -j
```

默认使用 `third-party/onnxruntime-genai/build/Linux/Release`；如需使用外部
构建，可通过 `-DORT_GENAI_ROOT=... -DORT_GENAI_BUILD=...` 覆盖。

### 7.3 运行

```bash
./build/qwen3_infer \
  --prompt "用三句话介绍一下北京" \
  --no-think \
  --max-new-tokens 128
```

指定非默认模型目录：

```bash
./build/qwen3_infer \
  --model /path/to/Qwen3-0.6B-onnx-genai/cpu_and_mobile/cpu-int4-rtn-block-32 \
  --prompt "你好" \
  --no-think
```

启用采样：

```bash
./build/qwen3_infer \
  --prompt "写一个关于月亮的短故事" \
  --sample \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --max-new-tokens 128
```

Qwen3-0.6B 默认启用思考模式，`--no-think` 切换为非思考模式；
`--raw-prompt` 可绕过 Chat Template，执行文本续写。

---

## 8. 工具：检查 ONNX 模型结构

`inspect_onnx` 输出 ONNX 图的输入输出名称、类型和维度，便于核对模型接口：

```bash
./build/inspect_onnx \
  /path/to/Qwen3.5-0.8B-ONNX-OPT/onnx/decoder_model_merged_q4.onnx
```
