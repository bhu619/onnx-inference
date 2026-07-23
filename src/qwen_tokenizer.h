#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class QwenTokenizer {
 public:
  explicit QwenTokenizer(const std::string& tokenizer_json);

  std::vector<int64_t> Encode(const std::string& text) const;
  std::string DecodeToken(int64_t id, bool skip_special = true) const;

 private:
  struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& value) const;
  };

  std::vector<std::string> Pretokenize(const std::string& text) const;
  std::vector<std::string> Bpe(const std::string& piece) const;
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
