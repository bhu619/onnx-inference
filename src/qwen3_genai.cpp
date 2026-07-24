// Qwen3-0.6B inference on ONNX Runtime GenAI: chat template through the
// GenAI tokenizer API and generator-driven decoding with streaming output.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ort_genai.h"

#include "qwen3_genai.h"
#include "terminal_ui.h"

namespace fs = std::filesystem;

// Command-line options; see Usage() for the accepted flags and defaults.
struct Options {
  std::string model;
  std::string prompt;
  std::string system = "You are a helpful assistant.";
  int max_new_tokens = 256;
  bool sample = false;
  double temperature = 0.6;
  double top_p = 0.95;
  int top_k = 20;
  double repetition_penalty = 1.0;
  bool raw_prompt = false;
  bool think = true;
};

[[noreturn]] void Usage(const char* program, const std::string& error = {}) {
  if (!error.empty()) std::cerr << "Error: " << error << "\n\n";
  std::cerr
      << "Usage: " << program << " --prompt TEXT [options]\n\n"
      << "Options:\n"
      << "  --model DIR                ORT GenAI model directory "
         "(default: ~/.cache/models/Qwen3-0.6B-onnx-genai/"
         "cpu_and_mobile/cpu-int4-rtn-block-32)\n"
      << "  --prompt TEXT              Initial prompt for the interactive session\n"
      << "  --system TEXT              System prompt\n"
      << "  --max-new-tokens N         Maximum generated tokens (default: 256)\n"
      << "  --sample                   Enable sampling (greedy by default)\n"
      << "  --temperature N            Sampling temperature (default: 0.6)\n"
      << "  --top-p N                  Nucleus sampling probability (default: 0.95)\n"
      << "  --top-k N                  Top-k sampling (default: 20)\n"
      << "  --repetition-penalty N     Repetition penalty (default: 1.0)\n"
      << "  --no-think                 Disable Qwen3's thinking mode\n"
      << "  --raw-prompt               Do not apply Qwen's chat template\n"
      << "  -h, --help                 Show this help\n";
  std::exit(error.empty() ? 0 : 2);
}

std::string NextValue(int& i, int argc, char** argv, const std::string& name) {
  if (++i >= argc) Usage(argv[0], name + " requires a value");
  return argv[i];
}

Options ParseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") Usage(argv[0]);
    if (arg == "--model") o.model = NextValue(i, argc, argv, arg);
    else if (arg == "--prompt") o.prompt = NextValue(i, argc, argv, arg);
    else if (arg == "--system") o.system = NextValue(i, argc, argv, arg);
    else if (arg == "--max-new-tokens") o.max_new_tokens = std::stoi(NextValue(i, argc, argv, arg));
    else if (arg == "--temperature") o.temperature = std::stod(NextValue(i, argc, argv, arg));
    else if (arg == "--top-p") o.top_p = std::stod(NextValue(i, argc, argv, arg));
    else if (arg == "--top-k") o.top_k = std::stoi(NextValue(i, argc, argv, arg));
    else if (arg == "--repetition-penalty") o.repetition_penalty = std::stod(NextValue(i, argc, argv, arg));
    else if (arg == "--sample") o.sample = true;
    else if (arg == "--no-think") o.think = false;
    else if (arg == "--raw-prompt") o.raw_prompt = true;
    else Usage(argv[0], "unknown option: " + arg);
  }
  if (o.max_new_tokens <= 0) Usage(argv[0], "--max-new-tokens must be positive");
  if (o.model.empty()) {
    o.model = terminal_ui::ModelCachePath(
        "Qwen3-0.6B-onnx-genai/cpu_and_mobile/cpu-int4-rtn-block-32").string();
  }
  return o;
}

std::string ReadFile(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path.string());
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

// Minimal JSON string escaping for the chat messages payload.
std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '\"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
              << std::dec << std::setfill(' ');
        } else {
          out << c;
        }
    }
  }
  return out.str();
}

// Applies the model's chat template unless --raw-prompt was given.
std::string MakePrompt(const Options& options, const OgaTokenizer& tokenizer) {
  if (options.raw_prompt) return options.prompt;
  const auto template_text = ReadFile(fs::path(options.model) / "chat_template.jinja");
  const std::string messages =
      "[{\"role\":\"system\",\"content\":\"" + JsonEscape(options.system) +
      "\"},{\"role\":\"user\",\"content\":\"" + JsonEscape(options.prompt) + "\"}]";
  std::string prompt = static_cast<const char*>(
      tokenizer.ApplyChatTemplate(template_text.c_str(), messages.c_str(), nullptr, true));
  // ORT GenAI's chat-template API does not expose arbitrary Jinja kwargs. This
  // is exactly what Qwen3's template emits for enable_thinking=false.
  if (!options.think) prompt += "<think>\n\n</think>\n\n";
  return prompt;
}

int run_qwen3_genai(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    terminal_ui::InstallInterruptHandler();
    if (!fs::is_regular_file(fs::path(options.model) / "genai_config.json")) {
      throw std::runtime_error(
          "invalid ONNX Runtime GenAI model directory: " + options.model +
          "/genai_config.json is missing; download the cpu-int4-rtn-block-32 "
          "variant from xiaoyao9184/Qwen3-0.6B-onnx-genai");
    }

    OgaHandle oga_handle;
    terminal_ui::PrintLoadingMessage();
    auto model = OgaModel::Create(options.model.c_str());
    auto tokenizer = OgaTokenizer::Create(*model);
    terminal_ui::PrintChatHeader(options.model);
    Options request = options;
    while (true) {
      if (!terminal_ui::ReadPrompt(request.prompt)) break;

      auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);
      const std::string prompt = MakePrompt(request, *tokenizer);
      auto input = OgaSequences::Create();
      tokenizer->Encode(prompt.c_str(), *input);
      const size_t prompt_tokens = input->SequenceCount(0);

      // Map CLI sampling options onto GenAI search options.
      auto params = OgaGeneratorParams::Create(*model);
      params->SetSearchOption("max_length", static_cast<double>(prompt_tokens + request.max_new_tokens));
      params->SetSearchOption("batch_size", 1);
      params->SetSearchOptionBool("do_sample", request.sample);
      params->SetSearchOption("temperature", request.temperature);
      params->SetSearchOption("top_p", request.top_p);
      params->SetSearchOption("top_k", request.top_k);
      params->SetSearchOption("repetition_penalty", request.repetition_penalty);

      auto generator = OgaGenerator::Create(*model, *params);
      terminal_ui::TerminalOutput output(request.think && !request.raw_prompt);
      terminal_ui::GenerationGuard generation;
      const auto started = std::chrono::steady_clock::now();
      generator->AppendTokenSequences(*input);

      auto first_token = started;
      size_t generated = 0;
      while (!generator->IsDone() && !generation.Interrupted()) {
        generator->GenerateNextToken();
        if (generated++ == 0) first_token = std::chrono::steady_clock::now();
        output.Write(tokenizer_stream->Decode(generator->GetNextTokens()[0]));
      }
      generation.Finish();
      const auto finished = std::chrono::steady_clock::now();
      const double prompt_seconds =
          std::chrono::duration<double>(first_token - started).count();
      const double generation_seconds =
          std::chrono::duration<double>(finished - first_token).count();
      output.Finish();
      terminal_ui::PrintStats(
          prompt_tokens, generated, prompt_seconds,
          prompt_seconds > 0 ? prompt_tokens / prompt_seconds : 0.0,
          generation_seconds > 0 ? generated / generation_seconds : 0.0);
      request.prompt.clear();
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Inference failed: " << e.what() << '\n';
    return 1;
  }
}
