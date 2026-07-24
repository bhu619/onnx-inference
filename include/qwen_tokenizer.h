#pragma once

// ByteLevel BPE tokenizer for Qwen models, loaded from a Hugging Face
// tokenizer.json. Encoding splits text around added (special) tokens, runs
// the Qwen pretokenizer on ordinary spans, and applies BPE merges over
// byte-level symbols; decoding maps symbols back to raw UTF-8 bytes.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class QwenTokenizer {
 public:
  // Loads vocab, BPE merges, and added tokens from tokenizer.json.
  explicit QwenTokenizer(const std::string& tokenizer_json);

  // Encodes text into token ids; added tokens are matched atomically.
  std::vector<int64_t> Encode(const std::string& text) const;
  // Decodes a single token id. Special tokens decode to their literal text,
  // or to "" when skip_special is set and the token is marked special.
  std::string DecodeToken(int64_t id, bool skip_special = true) const;

 private:
  struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& value) const;
  };

  // Splits ordinary text into pretoken pieces (words, numbers, symbol runs).
  std::vector<std::string> Pretokenize(const std::string& text) const;
  // Applies BPE merges to one pretokenized piece.
  std::vector<std::string> Bpe(const std::string& piece) const;
  // Maps raw bytes to byte-level symbols (GPT-2 style byte-to-unicode).
  std::vector<std::string> ByteEncode(const std::string& text) const;

  std::unordered_map<std::string, int64_t> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> merge_rank_;
  std::unordered_map<std::string, int64_t> special_to_id_;
  std::unordered_map<int64_t, std::string> id_to_special_;
  std::unordered_set<int64_t> skip_special_ids_;
  std::vector<std::string> specials_by_length_;
  std::vector<std::string> byte_to_symbol_;
  std::unordered_map<std::string, unsigned char> symbol_to_byte_;
};
