// Qwen3.5-0.8B inference on the ONNX Runtime C++ API: ByteLevel BPE
// tokenization, chat template, embedding + decoder graph execution with
// recurrent state and KV cache, and greedy or top-k/top-p sampling.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "qwen3_5_ort.h"
#include "qwen_image_processor.h"
#include "qwen_tokenizer.h"
#include "terminal_ui.h"

namespace fs = std::filesystem;

// Special token ids and vocabulary size of Qwen3.5-0.8B; must match the model.
constexpr int64_t kEos = 248044;
constexpr int64_t kImEnd = 248046;
constexpr int64_t kImagePad = 248056;
constexpr size_t kVocabSize = 248320;
constexpr size_t kHiddenSize = 1024;

// Command-line options; see Usage() for the accepted flags and defaults.
struct Options {
  fs::path model;
  fs::path image;
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
            << "  --model DIR              Model root (default: ~/.cache/models/Qwen3.5-0.8B-ONNX-OPT)\n"
            << "  --image FILE             Image to include with each prompt\n"
            << "  --prompt TEXT            Initial prompt for the interactive session\n"
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
    else if (a == "--image") o.image = Next(i, argc, argv);
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
  if (o.model.empty()) {
    o.model = terminal_ui::ModelCachePath("Qwen3.5-0.8B-ONNX-OPT");
  }
  if (o.max_new_tokens <= 0 || o.top_k <= 0 || o.temperature <= 0 || o.top_p <= 0 || o.top_p > 1)
    Usage(argv[0], "invalid generation option");
  return o;
}

// Builds the chat prompt manually: system and user turns, then an assistant
// opener. An immediately closed <think> block disables thinking mode.
std::string ChatPrompt(const Options& o, int64_t image_tokens) {
  std::string prompt = "<|im_start|>system\n" + o.system + "<|im_end|>\n";
  prompt += "<|im_start|>user\n";
  if (image_tokens > 0) {
    prompt += "<|vision_start|>";
    for (int64_t i = 0; i < image_tokens; ++i) prompt += "<|image_pad|>";
    prompt += "<|vision_end|>";
  }
  prompt += o.prompt + "<|im_end|>\n";
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

struct VisionContext {
  int64_t grid_t;
  int64_t grid_h;
  int64_t grid_w;
  std::vector<float> features;

  int64_t NumFeatures() const {
    return grid_t * (grid_h / 2) * (grid_w / 2);
  }
};

VisionContext EncodeImage(Ort::Session& vision, const fs::path& image_path) {
  QwenProcessedImage image = ProcessQwenImage(image_path);
  Ort::MemoryInfo cpu =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> pixel_shape{image.NumPatches(), 1536};
  std::vector<int64_t> grid_shape{1, 3};
  std::vector<int64_t> grid{image.grid_t, image.grid_h, image.grid_w};
  std::vector<Ort::Value> inputs;
  inputs.push_back(Ort::Value::CreateTensor<float>(
      cpu, image.pixel_values.data(), image.pixel_values.size(),
      pixel_shape.data(), pixel_shape.size()));
  inputs.push_back(Ort::Value::CreateTensor<int64_t>(
      cpu, grid.data(), grid.size(), grid_shape.data(), grid_shape.size()));
  auto input_names = Names(vision, true);
  auto output_names = Names(vision, false);
  auto input_ptrs = Pointers(input_names);
  auto output_ptrs = Pointers(output_names);
  auto outputs = vision.Run(Ort::RunOptions{nullptr}, input_ptrs.data(),
                            inputs.data(), inputs.size(), output_ptrs.data(),
                            output_ptrs.size());
  if (outputs.size() != 1) {
    throw std::runtime_error("vision encoder returned an unexpected output count");
  }
  const size_t expected =
      static_cast<size_t>(image.NumFeatures()) * kHiddenSize;
  const size_t actual =
      outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
  if (actual != expected) {
    throw std::runtime_error("vision encoder returned an unexpected feature shape");
  }
  const float* data = outputs[0].GetTensorData<float>();
  return {image.grid_t, image.grid_h, image.grid_w,
          std::vector<float>(data, data + actual)};
}

struct PositionLayout {
  std::vector<int64_t> values;
  int64_t next_position;
};

PositionLayout BuildPositionIds(const std::vector<int64_t>& ids,
                                const std::optional<VisionContext>& vision) {
  const size_t sequence = ids.size();
  PositionLayout layout{{}, static_cast<int64_t>(sequence)};
  layout.values.resize(3 * sequence);
  auto set_position = [&](size_t index, int64_t temporal, int64_t height,
                          int64_t width) {
    layout.values[index] = temporal;
    layout.values[sequence + index] = height;
    layout.values[2 * sequence + index] = width;
  };

  if (!vision) {
    for (size_t i = 0; i < sequence; ++i) {
      const int64_t position = static_cast<int64_t>(i);
      set_position(i, position, position, position);
    }
    return layout;
  }

  const auto first_pad = std::find(ids.begin(), ids.end(), kImagePad);
  if (first_pad == ids.end()) {
    throw std::runtime_error("image prompt does not contain image placeholder tokens");
  }
  const size_t pad_begin =
      static_cast<size_t>(std::distance(ids.begin(), first_pad));
  const size_t feature_count = static_cast<size_t>(vision->NumFeatures());
  const size_t pad_end = pad_begin + feature_count;
  if (pad_end > sequence ||
      !std::all_of(ids.begin() + static_cast<std::ptrdiff_t>(pad_begin),
                   ids.begin() + static_cast<std::ptrdiff_t>(pad_end),
                   [](int64_t id) { return id == kImagePad; }) ||
      (pad_end < sequence && ids[pad_end] == kImagePad)) {
    throw std::runtime_error("image placeholder count does not match vision features");
  }

  for (size_t i = 0; i < pad_begin; ++i) {
    const int64_t position = static_cast<int64_t>(i);
    set_position(i, position, position, position);
  }

  const int64_t merged_h = vision->grid_h / 2;
  const int64_t merged_w = vision->grid_w / 2;
  for (size_t i = 0; i < feature_count; ++i) {
    const int64_t index = static_cast<int64_t>(i);
    const int64_t temporal = index / (merged_h * merged_w);
    const int64_t spatial = index % (merged_h * merged_w);
    set_position(pad_begin + i, static_cast<int64_t>(pad_begin) + temporal,
                 static_cast<int64_t>(pad_begin) + spatial / merged_w,
                 static_cast<int64_t>(pad_begin) + spatial % merged_w);
  }

  const int64_t vision_span =
      std::max({vision->grid_t, merged_h, merged_w});
  const int64_t text_start = static_cast<int64_t>(pad_begin) + vision_span;
  for (size_t i = pad_end; i < sequence; ++i) {
    const int64_t position =
        text_start + static_cast<int64_t>(i - pad_end);
    set_position(i, position, position, position);
  }
  layout.next_position =
      text_start + static_cast<int64_t>(sequence - pad_end);
  return layout;
}

int run_qwen3_5_ort(int argc, char** argv) {
  try {
    const Options options = ParseArgs(argc, argv);
    terminal_ui::InstallInterruptHandler();
    const fs::path embed_path = options.model / "onnx/embed_tokens_q4.onnx";
    const fs::path decoder_path = options.model / "onnx/decoder_model_merged_q4.onnx";
    const fs::path vision_path = options.model / "onnx/vision_encoder_q4.onnx";
    for (const auto& path : {embed_path, decoder_path, options.model / "tokenizer.json"})
      if (!fs::is_regular_file(path)) throw std::runtime_error("missing file: " + path.string());
    if (!options.image.empty() && !fs::is_regular_file(vision_path)) {
      throw std::runtime_error("missing file: " + vision_path.string());
    }

    // Load the tokenizer and ONNX graphs. The vision graph is optional.
    terminal_ui::PrintLoadingMessage();
    QwenTokenizer tokenizer((options.model / "tokenizer.json").string());
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "qwen3_5-ort");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetLogSeverityLevel(3);
    if (options.intra_threads > 0) session_options.SetIntraOpNumThreads(options.intra_threads);
    Ort::Session embed(env, embed_path.c_str(), session_options);
    Ort::Session decoder(env, decoder_path.c_str(), session_options);
    std::unique_ptr<Ort::Session> vision;
    std::optional<VisionContext> vision_context;
    if (!options.image.empty()) {
      vision =
          std::make_unique<Ort::Session>(env, vision_path.c_str(), session_options);
      vision_context = EncodeImage(*vision, options.image);
    }
    auto embed_inputs = Names(embed, true), embed_outputs = Names(embed, false);
    auto decoder_inputs = Names(decoder, true), decoder_outputs = Names(decoder, false);
    auto embed_input_ptrs = Pointers(embed_inputs), embed_output_ptrs = Pointers(embed_outputs);
    auto decoder_input_ptrs = Pointers(decoder_inputs), decoder_output_ptrs = Pointers(decoder_outputs);
    terminal_ui::PrintChatHeader(options.model);

    Options request = options;
    while (true) {
      if (!terminal_ui::ReadPrompt(request.prompt)) break;

      const int64_t image_tokens =
          vision_context ? vision_context->NumFeatures() : 0;
      std::vector<int64_t> current_ids =
          tokenizer.Encode(ChatPrompt(request, image_tokens));
      const size_t prompt_tokens = current_ids.size();
      if (current_ids.empty()) throw std::runtime_error("tokenizer returned an empty prompt");
      const PositionLayout prompt_positions =
          BuildPositionIds(current_ids, vision_context);
      int64_t next_position = prompt_positions.next_position;
      terminal_ui::TerminalOutput output(request.think);

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
      std::mt19937 random(request.seed);
      const auto started = std::chrono::steady_clock::now();
      auto first_token_time = started;
      size_t generated = 0;
      terminal_ui::GenerationGuard generation;
      // Prefill the full prompt, then decode one token per iteration.
      for (; generated < static_cast<size_t>(request.max_new_tokens) &&
             !generation.Interrupted(); ++generated) {
        std::vector<int64_t> ids_shape{1, static_cast<int64_t>(current_ids.size())};
        auto ids_tensor = Ort::Value::CreateTensor<int64_t>(cpu, current_ids.data(), current_ids.size(),
                                                            ids_shape.data(), ids_shape.size());
        auto embedded = embed.Run(Ort::RunOptions{nullptr}, embed_input_ptrs.data(), &ids_tensor, 1,
                                  embed_output_ptrs.data(), 1);
        if (past_length == 0 && vision_context) {
          float* embeddings = embedded[0].GetTensorMutableData<float>();
          size_t feature = 0;
          for (size_t i = 0; i < current_ids.size(); ++i) {
            if (current_ids[i] != kImagePad) continue;
            std::copy_n(vision_context->features.data() + feature * kHiddenSize,
                        kHiddenSize, embeddings + i * kHiddenSize);
            ++feature;
          }
          if (feature != static_cast<size_t>(vision_context->NumFeatures())) {
            throw std::runtime_error(
                "failed to place all image features in prompt embeddings");
          }
        }

        const int64_t sequence = static_cast<int64_t>(current_ids.size());
        std::vector<int64_t> mask(static_cast<size_t>(past_length + sequence), 1);
        std::vector<int64_t> positions;
        if (past_length == 0) {
          positions = prompt_positions.values;
        } else {
          positions.resize(static_cast<size_t>(3 * sequence));
          for (int axis = 0; axis < 3; ++axis) {
            for (int64_t i = 0; i < sequence; ++i) {
              positions[static_cast<size_t>(axis * sequence + i)] =
                  next_position + i;
            }
          }
          next_position += sequence;
        }
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

        int64_t token = SelectToken(outputs[0].GetTensorMutableData<float>(), request, seen, random);
        if (generated == 0) first_token_time = std::chrono::steady_clock::now();
        if (token == kEos || token == kImEnd) break;
        seen.insert(token);
        output.Write(tokenizer.DecodeToken(token));
        current_ids.assign(1, token);
      }
      generation.Finish();
      const auto finished = std::chrono::steady_clock::now();
      const double prompt_seconds =
          std::chrono::duration<double>(first_token_time - started).count();
      const double generation_seconds =
          std::chrono::duration<double>(finished - first_token_time).count();
      output.Finish();
      terminal_ui::PrintStats(
          prompt_tokens, generated, prompt_seconds,
          prompt_seconds > 0 ? prompt_tokens / prompt_seconds : 0.0,
          generation_seconds > 0 ? generated / generation_seconds : 0.0);
      request.prompt.clear();
    }
    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << '\n';
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
