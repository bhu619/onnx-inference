// Qwen3.5-0.8B inference on the ONNX Runtime C++ API: ByteLevel BPE
// tokenization, chat template, embedding + decoder graph execution with
// recurrent state and KV cache, and greedy or top-k/top-p sampling.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "qwen3_5_ort.h"
#include "qwen_tokenizer.h"

namespace fs = std::filesystem;

// Special token ids and vocabulary size of Qwen3.5-0.8B; must match the model.
constexpr int64_t kEos = 248044;
constexpr int64_t kImEnd = 248046;
constexpr size_t kVocabSize = 248320;

// Command-line options; see Usage() for the accepted flags and defaults.
struct Options {
  fs::path model = "/home/ubuntu/.cache/models/Qwen3.5-0.8B-ONNX-OPT";
  std::string prompt;
  std::string system = "You are a helpful assistant.";
  int max_new_tokens = 128;
  int intra_threads = 0;
  bool sample = false;
  bool think = false;
  double temperature = 0.6;
  int top_k = 20;
  double top_p = 0.95;
  double presence_penalty = 0.0;
  uint32_t seed = 42;
};

[[noreturn]] void Usage(const char* exe, const std::string& error = {}) {
  if (!error.empty()) std::cerr << "Error: " << error << "\n\n";
  std::cerr << "Usage: " << exe << " --prompt TEXT [options]\n\n"
            << "  --model DIR              Model root (default: Qwen3.5-0.8B-ONNX-OPT cache path)\n"
            << "  --prompt TEXT            User prompt; reads one stdin line if omitted\n"
            << "  --system TEXT            System message\n"
            << "  --max-new-tokens N       Generation limit (default: 128)\n"
            << "  --threads N              ORT intra-op threads (default: ORT chooses)\n"
            << "  --sample                 Enable top-k/top-p sampling (greedy by default)\n"
            << "  --temperature N          Sampling temperature (default: 0.6)\n"
            << "  --top-k N                Sampling top-k (default: 20)\n"
            << "  --top-p N                Sampling top-p (default: 0.95)\n"
            << "  --presence-penalty N     Penalize tokens already generated\n"
            << "  --seed N                 Sampling seed (default: 42)\n"
            << "  --think                  Enable Qwen3.5 thinking mode\n"
            << "  -h, --help               Show help\n";
  std::exit(error.empty() ? 0 : 2);
}

std::string Next(int& i, int argc, char** argv) {
  if (++i >= argc) Usage(argv[0], "missing option value");
  return argv[i];
}

Options ParseArgs(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") Usage(argv[0]);
    else if (a == "--model") o.model = Next(i, argc, argv);
    else if (a == "--prompt") o.prompt = Next(i, argc, argv);
    else if (a == "--system") o.system = Next(i, argc, argv);
    else if (a == "--max-new-tokens") o.max_new_tokens = std::stoi(Next(i, argc, argv));
    else if (a == "--threads") o.intra_threads = std::stoi(Next(i, argc, argv));
    else if (a == "--temperature") o.temperature = std::stod(Next(i, argc, argv));
    else if (a == "--top-k") o.top_k = std::stoi(Next(i, argc, argv));
    else if (a == "--top-p") o.top_p = std::stod(Next(i, argc, argv));
    else if (a == "--presence-penalty") o.presence_penalty = std::stod(Next(i, argc, argv));
    else if (a == "--seed") o.seed = static_cast<uint32_t>(std::stoul(Next(i, argc, argv)));
    else if (a == "--sample") o.sample = true;
    else if (a == "--think") o.think = true;
    else Usage(argv[0], "unknown option: " + a);
  }
  if (o.prompt.empty()) std::getline(std::cin, o.prompt);
  if (o.prompt.empty()) Usage(argv[0], "prompt must not be empty");
  if (o.max_new_tokens <= 0 || o.top_k <= 0 || o.temperature <= 0 || o.top_p <= 0 || o.top_p > 1)
    Usage(argv[0], "invalid generation option");
  return o;
}

// Builds the chat prompt manually: system and user turns, then an assistant
// opener. An immediately closed <think> block disables thinking mode.
std::string ChatPrompt(const Options& o) {
  std::string prompt = "<|im_start|>system\n" + o.system + "<|im_end|>\n";
  prompt += "<|im_start|>user\n" + o.prompt + "<|im_end|>\n";
  prompt += "<|im_start|>assistant\n";
  prompt += o.think ? "<think>\n" : "<think>\n\n</think>\n\n";
  return prompt;
}

// Returns the input or output names of a session, in graph order.
std::vector<std::string> Names(Ort::Session& session, bool inputs) {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
  std::vector<std::string> names;
  for (size_t i = 0; i < count; ++i) {
    auto name = inputs ? session.GetInputNameAllocated(i, allocator)
                       : session.GetOutputNameAllocated(i, allocator);
    names.emplace_back(name.get());
  }
  return names;
}

// Stable C-string views over a name list, as required by Ort::Session::Run.
std::vector<const char*> Pointers(const std::vector<std::string>& names) {
  std::vector<const char*> result;
  for (const auto& name : names) result.push_back(name.c_str());
  return result;
}

// Allocates a zero-filled float tensor; used for the initial decoder states.
Ort::Value ZeroTensor(Ort::AllocatorWithDefaultOptions& allocator,
                      const std::vector<int64_t>& shape) {
  auto value = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  const size_t count = value.GetTensorTypeAndShapeInfo().GetElementCount();
  if (count) std::fill_n(value.GetTensorMutableData<float>(), count, 0.0f);
  return value;
}

// Selects the next token from the logits: greedy argmax, or top-k filtering
// followed by a temperature softmax and top-p (nucleus) truncation. The
// presence penalty subtracts a fixed value from tokens already generated.
int64_t SelectToken(float* logits, const Options& o, const std::unordered_set<int64_t>& seen,
                    std::mt19937& random) {
  if (o.presence_penalty != 0) {
    for (int64_t id : seen) if (id >= 0 && static_cast<size_t>(id) < kVocabSize) logits[id] -= o.presence_penalty;
  }
  if (!o.sample) return static_cast<int64_t>(std::max_element(logits, logits + kVocabSize) - logits);

  const size_t k = std::min<size_t>(static_cast<size_t>(o.top_k), kVocabSize);
  std::vector<int64_t> ids(kVocabSize);
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(k), ids.end(),
                    [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });
  ids.resize(k);
  const float maximum = logits[ids[0]];
  std::vector<double> probabilities(k);
  double sum = 0;
  for (size_t i = 0; i < k; ++i) sum += probabilities[i] = std::exp((logits[ids[i]] - maximum) / o.temperature);
  for (double& p : probabilities) p /= sum;
  double cumulative = 0;
  size_t keep = 0;
  do { cumulative += probabilities[keep++]; } while (keep < k && cumulative < o.top_p);
  probabilities.resize(keep);
  ids.resize(keep);
  return ids[std::discrete_distribution<size_t>(probabilities.begin(), probabilities.end())(random)];
}

int run_qwen3_5_ort(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    const fs::path embed_path = options.model / "onnx/embed_tokens_q4.onnx";
    const fs::path decoder_path = options.model / "onnx/decoder_model_merged_q4.onnx";
    for (const auto& path : {embed_path, decoder_path, options.model / "tokenizer.json"})
      if (!fs::is_regular_file(path)) throw std::runtime_error("missing file: " + path.string());

    // Load the tokenizer and both ONNX graphs (embedding + merged decoder).
    std::cerr << "Loading tokenizer and ONNX sessions...\n";
    QwenTokenizer tokenizer((options.model / "tokenizer.json").string());
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "qwen3_5-ort");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (options.intra_threads > 0) session_options.SetIntraOpNumThreads(options.intra_threads);
    Ort::Session embed(env, embed_path.c_str(), session_options);
    Ort::Session decoder(env, decoder_path.c_str(), session_options);
    auto embed_inputs = Names(embed, true), embed_outputs = Names(embed, false);
    auto decoder_inputs = Names(decoder, true), decoder_outputs = Names(decoder, false);
    auto embed_input_ptrs = Pointers(embed_inputs), embed_output_ptrs = Pointers(embed_outputs);
    auto decoder_input_ptrs = Pointers(decoder_inputs), decoder_output_ptrs = Pointers(decoder_outputs);

    std::vector<int64_t> current_ids = tokenizer.Encode(ChatPrompt(options));
    const size_t prompt_tokens = current_ids.size();
    if (current_ids.empty()) throw std::runtime_error("tokenizer returned an empty prompt");
    std::cerr << "Prompt tokens: " << prompt_tokens << "\nOutput: " << std::flush;

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Initial decoder states, aligned with the graph inputs that follow
    // inputs_embeds / attention_mask / position_ids / keep_logits:
    //   past_conv.*      -> {1, 6144, 3}      (short conv state)
    //   past_recurrent.* -> {1, 16, 128, 128} (recurrent state)
    //   remaining        -> {1, 2, 0, 256}    (empty KV cache)
    std::vector<Ort::Value> states;
    for (size_t i = 4; i < decoder_inputs.size(); ++i) {
      const std::string& name = decoder_inputs[i];
      if (name.rfind("past_conv.", 0) == 0) states.push_back(ZeroTensor(allocator, {1, 6144, 3}));
      else if (name.rfind("past_recurrent.", 0) == 0) states.push_back(ZeroTensor(allocator, {1, 16, 128, 128}));
      else states.push_back(ZeroTensor(allocator, {1, 2, 0, 256}));
    }

    int64_t past_length = 0;
    std::unordered_set<int64_t> seen;
    std::mt19937 random(options.seed);
    const auto started = std::chrono::steady_clock::now();
    auto first_token_time = started;
    size_t generated = 0;
    // Prefill the full prompt, then decode one token per iteration.
    for (; generated < static_cast<size_t>(options.max_new_tokens); ++generated) {
      std::vector<int64_t> ids_shape{1, static_cast<int64_t>(current_ids.size())};
      auto ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu, current_ids.data(), current_ids.size(),
                                                          ids_shape.data(), ids_shape.size());
      auto embedded = embed.Run(Ort::RunOptions{nullptr}, embed_input_ptrs.data(), &ids_tensor, 1,
                                embed_output_ptrs.data(), 1);

      const int64_t sequence = static_cast<int64_t>(current_ids.size());
      std::vector<int64_t> mask(static_cast<size_t>(past_length + sequence), 1);
      // MRoPE position_ids with shape [3, 1, seq]; for text, all three axes
      // carry the same ascending positions.
      std::vector<int64_t> positions(static_cast<size_t>(3 * sequence));
      for (int axis = 0; axis < 3; ++axis)
        for (int64_t i = 0; i < sequence; ++i) positions[static_cast<size_t>(axis * sequence + i)] = past_length + i;
      int64_t keep_logits = 1;
      std::vector<int64_t> mask_shape{1, past_length + sequence};
      std::vector<int64_t> position_shape{3, 1, sequence};
      std::vector<int64_t> scalar_shape;

      std::vector<Ort::Value> inputs;
      inputs.reserve(decoder_inputs.size());
      inputs.push_back(std::move(embedded[0]));
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(cpu, mask.data(), mask.size(), mask_shape.data(), mask_shape.size()));
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(cpu, positions.data(), positions.size(), position_shape.data(), position_shape.size()));
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(cpu, &keep_logits, 1, scalar_shape.data(), 0));
      for (auto& state : states) inputs.push_back(std::move(state));

      auto outputs = decoder.Run(Ort::RunOptions{nullptr}, decoder_input_ptrs.data(), inputs.data(), inputs.size(),
                                 decoder_output_ptrs.data(), decoder_output_ptrs.size());
      past_length += sequence;
      // Outputs after the logits become the next step's recurrent/KV states.
      states.clear();
      states.reserve(outputs.size() - 1);
      for (size_t i = 1; i < outputs.size(); ++i) states.push_back(std::move(outputs[i]));

      int64_t token = SelectToken(outputs[0].GetTensorMutableData<float>(), options, seen, random);
      if (generated == 0) first_token_time = std::chrono::steady_clock::now();
      if (token == kEos || token == kImEnd) break;
      seen.insert(token);
      std::cout << tokenizer.DecodeToken(token) << std::flush;
      current_ids.assign(1, token);
    }
    const auto finished = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(finished - started).count();
    const double first = std::chrono::duration<double>(first_token_time - started).count();
    std::cout << "\n\n[prompt=" << prompt_tokens << " tokens, generated=" << generated
              << ", first-token=" << std::fixed << std::setprecision(3) << first
              << "s, throughput=" << (elapsed > 0 ? generated / elapsed : 0) << " token/s]\n";
    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << '\n';
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
