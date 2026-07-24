# onnx-inference

基于 [ONNX Runtime](https://github.com/microsoft/onnxruntime) 的 Qwen 系列模型 C++ 推理示例。不依赖 Python 与
Transformers，通过 ONNX Runtime C++ API 完成分词、Chat Template、图执行与采样解码的完整推理流程。

## 1. 示例程序

| 程序 | 目标模型 | 推理后端 | 默认构建 |
| --- | --- | --- | --- |
| `qwen3_5_ort` | Qwen3.5-0.8B | ONNX Runtime（C++ API） | 是 |
| `qwen3_infer` | Qwen3-0.6B | ONNX Runtime GenAI | 否（需显式启用） |
| `inspect_onnx` | — | ONNX Runtime（模型结构检查工具） | 是 |

两个推理示例相互独立，可单独构建和使用。

---

## 2. 功能特性

`qwen3_5_ort` 基于 ONNX Runtime C++ API 实现，不链接 ONNX Runtime GenAI，也不依赖 `genai_config.json`：

- Qwen ByteLevel BPE 分词与流式解码
- Chat Template 拼接（支持思考 / 非思考模式）
- Embedding 与 Decoder 两张 ONNX 图的执行循环
- 18 组 recurrent state 与 6 组 KV Cache 的维护
- MRoPE 三维 `position_ids`
- Greedy、top-k、top-p、temperature 与 presence penalty 解码策略
- EOS 检测与推理性能统计

当前示例仅支持纯文本推理，不调用 `vision_encoder_q4.onnx`。

---

## 3. 目录结构

```text
├── include/
│   ├── qwen3_5_ort.h           # Qwen3.5-0.8B 推理接口
│   ├── qwen3_genai.h           # Qwen3-0.6B 推理接口
│   ├── qwen_tokenizer.h        # Qwen BPE 分词器接口
│   └── terminal_ui.h           # 交互终端、分段着色与统计输出
├── src/
│   ├── main.cpp                # 共享入口（按编译宏分发到推理后端）
│   ├── qwen3_5_ort.cpp         # Qwen3.5-0.8B 推理（ONNX Runtime C++ API）
│   ├── qwen3_genai.cpp         # Qwen3-0.6B 推理（ONNX Runtime GenAI）
│   └── qwen_tokenizer.cpp      # Qwen BPE 分词器
├── tools/
│   └── inspect_onnx.cpp       # ONNX 模型输入输出检查工具
├── scripts/
│   └── init_submodules.sh     # 初始化 third-party Git 子模块
├── third-party/
│   ├── onnxruntime/           # Git 子模块，固定为 v1.27.1
│   └── onnxruntime-genai/     # Git 子模块，固定为 v0.14.0
└── CMakeLists.txt
```

---

## 4. 环境依赖

- CMake >= 3.18，支持 C++17 的编译器
- Git
- （可选）`hf` CLI（`pip install -U "huggingface_hub[cli]"`），仅用于下载模型

ONNX Runtime 与 ONNX Runtime GenAI 以 Git 子模块形式引入 `third-party/`，版本固定如下：

| 子模块 | 版本 | 简介 |
| --- | --- | --- |
| [`third-party/onnxruntime`](https://github.com/microsoft/onnxruntime) | `v1.27.1` | 跨平台 ONNX 推理引擎；`qwen3_5_ort` 与 `inspect_onnx` 直接调用其 C++ API |
| [`third-party/onnxruntime-genai`](https://github.com/microsoft/onnxruntime-genai) | `v0.14.0` | 基于 ONNX Runtime 的生成式推理库；`qwen3_infer` 依赖其 Chat Template 与生成接口 |

---

## 5. 获取源码

```bash
git clone https://github.com/bhu619/onnx-inference.git
cd onnx-inference
```

---

## 6. 初始化依赖

项目依赖通过 Git 子模块管理。选择以下任一方式完成初始化。

### 6.1 使用初始化脚本（推荐）

脚本会先同步 `.gitmodules` 中配置的子模块地址，再初始化所选子模块及其嵌套子模块。首次使用建议初始化全部依赖；主仓库更新子模块版本后，重新执行相同命令即可。

```bash
./scripts/init_submodules.sh --help                         # 查看完整参数说明

./scripts/init_submodules.sh --all                          # 初始化全部依赖
./scripts/init_submodules.sh --all --dry-run                # 预览将执行的 Git 命令

./scripts/init_submodules.sh -s onnxruntime                 # 仅初始化 ONNX Runtime
./scripts/init_submodules.sh -s onnxruntime-genai           # 仅初始化 ONNX Runtime GenAI
./scripts/init_submodules.sh -s onnxruntime,onnxruntime-genai  # 初始化多个指定子模块
```

`--submodule`（或 `-s`）可重复使用，也可传入逗号分隔的多个名称；不带参数运行脚本时同样会显示帮助。

### 6.2 使用 Git 命令

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

仅初始化单个子模块时追加其路径，例如：

```bash
git submodule update --init --recursive third-party/onnxruntime
```

初始化完成后可核对子模块版本：

```bash
git submodule status
```

---

## 7. 快速开始：Qwen3.5-0.8B（`qwen3_5_ort`）

### 7.1 准备模型文件

模型托管于 Hugging Face：
[onnx-community/Qwen3.5-0.8B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT/tree/main)。
使用 `hf` 下载：

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
  --local-dir "$HOME/.cache/models/Qwen3.5-0.8B-ONNX-OPT"
```

推理必需的文件如下（其余为可选配置与多模态文件）：

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

示例默认从 `~/.cache/models/Qwen3.5-0.8B-ONNX-OPT` 加载模型。模型存放在其他位置时，通过 `--model` 指定模型根目录。

### 7.2 编译

`qwen3_5_ort` 依赖 ONNX Runtime 共享库。请按照 ONNX Runtime 官方
[Build ONNX Runtime for inferencing](https://onnxruntime.ai/docs/build/inferencing.html)，在 `third-party/onnxruntime` 子模块中完成构建。

完成 ONNX Runtime 构建后，在仓库根目录配置并构建示例程序：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

构建产物为 `build/qwen3_5_ort`。默认配置使用子模块目录 `third-party/onnxruntime` 及其 `build/Linux/RelWithDebInfo` 构建输出；CMake 会写入运行时库搜索路径，因此运行时无需设置 `LD_LIBRARY_PATH`。

若 ONNX Runtime 位于其他位置，在配置时同时指定源码目录和包含 `libonnxruntime.so` 的构建目录：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DONNXRUNTIME_BUILD=/path/to/onnxruntime-build
```

### 7.3 运行

程序始终以交互模式运行。不指定 `--prompt` 时直接显示输入提示符：

```bash
./build/qwen3_5_ort \
  --think \
  --max-new-tokens 512 \
  --threads 4
```

基本推理（默认非思考模式）：

```bash
./build/qwen3_5_ort \
  --prompt "用三句话介绍一下北京" \
  --max-new-tokens 256 \
  --threads 4
```

启用思考模式：

```bash
./build/qwen3_5_ort \
  --prompt "hello, who are you?" \
  --think \
  --max-new-tokens 512 \
  --threads 4
```

采样解码（默认 Greedy，`--sample` 启用 top-k/top-p 采样）：

```bash
./build/qwen3_5_ort \
  --prompt "写一个关于月亮的短故事" \
  --sample \
  --temperature 0.6 \
  --top-k 20 \
  --top-p 0.95 \
  --presence-penalty 1.5 \
  --max-new-tokens 256 \
  --threads 4
```

### 7.4 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--model DIR` | 模型根目录 |
| `--prompt TEXT` | 交互会话的首条提示词 |
| `--system TEXT` | System 消息 |
| `--max-new-tokens N` | 最大生成 Token 数 |
| `--threads N` | ONNX Runtime CPU intra-op 线程数 |
| `--sample` | 启用 top-k/top-p 采样 |
| `--temperature N` | 采样温度 |
| `--top-k N` | top-k |
| `--top-p N` | top-p |
| `--presence-penalty N` | 对已生成的 Token 施加惩罚 |
| `--seed N` | 随机种子 |
| `--think` | 启用 Qwen3.5 思考模式 |
| `-h`, `--help` | 显示帮助 |

---

## 8. 可选示例：Qwen3-0.6B（`qwen3_infer`）

`qwen3_infer` 基于 ONNX Runtime GenAI，与 `qwen3_5_ort` 相互独立，默认不参与构建。

### 8.1 下载模型

使用 Hugging Face 上已转换的
[`xiaoyao9184/Qwen3-0.6B-onnx-genai`](https://huggingface.co/xiaoyao9184/Qwen3-0.6B-onnx-genai)，
其设备子目录包含 `genai_config.json`、ONNX 图、权重、Tokenizer 与 Chat Template，无需自行转换：

```bash
hf download xiaoyao9184/Qwen3-0.6B-onnx-genai \
  --include "cpu_and_mobile/cpu-int4-rtn-block-32/*" \
  --local-dir "$HOME/.cache/models/Qwen3-0.6B-onnx-genai"
```

下载后的目录结构：

```text
~/.cache/models/Qwen3-0.6B-onnx-genai/
└── cpu_and_mobile/
    └── cpu-int4-rtn-block-32/
        ├── chat_template.jinja
        ├── genai_config.json
        ├── model.onnx
        ├── model.onnx.data
        ├── tokenizer.json
        └── tokenizer_config.json
```

`qwen3_infer` 默认使用上述 CPU INT4 子目录。模型存放在其他位置或使用其他设备版本时，通过 `--model` 指向直接包含 `genai_config.json` 的子目录。

### 8.2 编译

`qwen3_infer` 依赖 ONNX Runtime GenAI 及其配套 ONNX Runtime。请按照 ONNX Runtime GenAI 官方
[Build the generate() API from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html)，在 `third-party/onnxruntime-genai` 子模块中完成构建。

完成 ONNX Runtime GenAI 构建后，启用 `BUILD_ORT_GENAI_EXAMPLE` 配置并构建示例程序：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_ORT_GENAI_EXAMPLE=ON

cmake --build build -j$(nproc)
```

GenAI 默认取自 `third-party/onnxruntime-genai/build/Linux/Release`，
可通过 `-DORT_GENAI_ROOT=... -DORT_GENAI_BUILD=...` 指向外部构建。

### 8.3 运行

程序始终以交互模式运行。不指定 `--prompt` 时直接显示输入提示符：

```bash
./build/qwen3_infer
```

指定首条提示词：

```bash
./build/qwen3_infer \
  --prompt "用三句话介绍一下北京" \
  --no-think \
  --max-new-tokens 256
```

指定其他模型目录：

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
  --max-new-tokens 512
```

Qwen3-0.6B 默认启用思考模式；`--no-think` 用于关闭思考模式，`--raw-prompt` 用于绕过 Chat Template。

---

## 9. 工具：检查 ONNX 模型结构

`inspect_onnx` 打印 ONNX 图的输入/输出名称、类型与维度，用于核对模型接口：

```bash
./build/inspect_onnx \
  /path/to/Qwen3.5-0.8B-ONNX-OPT/onnx/decoder_model_merged_q4.onnx
```
