#include "qwen_tokenizer.h"

#include <algorithm>
#include <clocale>
#include <cwctype>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace {

struct Rune {
  uint32_t codepoint;
  size_t begin;
  size_t end;
};

std::string Utf8(uint32_t cp) {
  std::string result;
  if (cp <= 0x7f) {
    result.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7ff) {
    result.push_back(static_cast<char>(0xc0 | (cp >> 6)));
    result.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else if (cp <= 0xffff) {
    result.push_back(static_cast<char>(0xe0 | (cp >> 12)));
    result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    result.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    result.push_back(static_cast<char>(0xf0 | (cp >> 18)));
    result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    result.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
  return result;
}

std::vector<Rune> DecodeUtf8(const std::string& text) {
  std::vector<Rune> out;
  for (size_t i = 0; i < text.size();) {
    const size_t begin = i;
    const auto first = static_cast<unsigned char>(text[i++]);
    uint32_t cp = 0;
    int continuation = 0;
    if (first < 0x80) cp = first;
    else if ((first & 0xe0) == 0xc0) { cp = first & 0x1f; continuation = 1; }
    else if ((first & 0xf0) == 0xe0) { cp = first & 0x0f; continuation = 2; }
    else if ((first & 0xf8) == 0xf0) { cp = first & 0x07; continuation = 3; }
    else throw std::runtime_error("invalid UTF-8 in prompt");
    for (int j = 0; j < continuation; ++j) {
      if (i >= text.size() || (static_cast<unsigned char>(text[i]) & 0xc0) != 0x80)
        throw std::runtime_error("invalid UTF-8 in prompt");
      cp = (cp << 6) | (static_cast<unsigned char>(text[i++]) & 0x3f);
    }
    out.push_back({cp, begin, i});
  }
  return out;
}

bool IsNewline(uint32_t c) { return c == '\r' || c == '\n'; }
bool IsSpace(uint32_t c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f' ||
         c == 0x85 || c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
         c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}
bool IsMark(uint32_t c) {
  return (c >= 0x0300 && c <= 0x036f) || (c >= 0x1ab0 && c <= 0x1aff) ||
         (c >= 0x1dc0 && c <= 0x1dff) || (c >= 0x20d0 && c <= 0x20ff) ||
         (c >= 0xfe20 && c <= 0xfe2f);
}
bool IsLetter(uint32_t c) {
  if (c < 128) return std::isalpha(static_cast<unsigned char>(c)) != 0;
  return std::iswalpha(static_cast<wint_t>(c)) != 0 ||
         (c >= 0x3400 && c <= 0x9fff) || (c >= 0x20000 && c <= 0x323af);
}
bool IsNumber(uint32_t c) {
  if (c < 128) return std::isdigit(static_cast<unsigned char>(c)) != 0;
  return std::iswdigit(static_cast<wint_t>(c)) != 0;
}
bool IsWord(uint32_t c) { return IsLetter(c) || IsMark(c) || IsNumber(c); }

}  // namespace

size_t QwenTokenizer::PairHash::operator()(
    const std::pair<std::string, std::string>& value) const {
  const size_t a = std::hash<std::string>{}(value.first);
  const size_t b = std::hash<std::string>{}(value.second);
  return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
}

QwenTokenizer::QwenTokenizer(const std::string& tokenizer_json) {
  std::setlocale(LC_CTYPE, "C.UTF-8");
  std::ifstream input(tokenizer_json);
  if (!input) throw std::runtime_error("cannot open tokenizer: " + tokenizer_json);
  nlohmann::json config;
  input >> config;
  if (config["model"].value("type", "") != "BPE")
    throw std::runtime_error("tokenizer model is not BPE");

  int64_t max_id = 0;
  for (auto it = config["model"]["vocab"].begin(); it != config["model"]["vocab"].end(); ++it) {
    const int64_t id = it.value().get<int64_t>();
    vocab_.emplace(it.key(), id);
    max_id = std::max(max_id, id);
  }
  for (const auto& token : config["added_tokens"]) {
    const int64_t id = token["id"].get<int64_t>();
    const std::string content = token["content"].get<std::string>();
    max_id = std::max(max_id, id);
    // Every added token is matched atomically. Some semantic control tokens,
    // notably <think>, deliberately have special=false in Qwen3.5.
    special_to_id_[content] = id;
    id_to_special_[id] = content;
    specials_by_length_.push_back(content);
    if (token.value("special", false)) skip_special_ids_.insert(id);
  }
  id_to_token_.resize(static_cast<size_t>(max_id + 1));
  for (const auto& item : vocab_) id_to_token_[static_cast<size_t>(item.second)] = item.first;
  std::sort(specials_by_length_.begin(), specials_by_length_.end(),
            [](const auto& a, const auto& b) { return a.size() > b.size(); });

  size_t rank = 0;
  for (const auto& merge : config["model"]["merges"]) {
    merge_rank_.emplace(std::make_pair(merge[0].get<std::string>(), merge[1].get<std::string>()), rank++);
  }

  std::vector<int> bytes;
  for (int c = '!'; c <= '~'; ++c) bytes.push_back(c);
  for (int c = 0xa1; c <= 0xac; ++c) bytes.push_back(c);
  for (int c = 0xae; c <= 0xff; ++c) bytes.push_back(c);
  std::vector<int> codepoints = bytes;
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) {
      bytes.push_back(b);
      codepoints.push_back(256 + extra++);
    }
  }
  byte_to_symbol_.resize(256);
  for (size_t i = 0; i < bytes.size(); ++i) {
    const std::string symbol = Utf8(static_cast<uint32_t>(codepoints[i]));
    byte_to_symbol_[static_cast<size_t>(bytes[i])] = symbol;
    symbol_to_byte_[symbol] = static_cast<unsigned char>(bytes[i]);
  }
}

std::vector<std::string> QwenTokenizer::Pretokenize(const std::string& text) const {
  const auto runes = DecodeUtf8(text);
  std::vector<std::string> pieces;
  size_t i = 0;
  auto emit = [&](size_t begin, size_t end) {
    pieces.push_back(text.substr(runes[begin].begin, runes[end - 1].end - runes[begin].begin));
  };
  while (i < runes.size()) {
    // Case-insensitive English contractions.
    if (runes[i].codepoint == '\'') {
      static const std::vector<std::string> suffixes = {"re", "ve", "ll", "s", "t", "m", "d"};
      bool matched = false;
      for (const auto& suffix : suffixes) {
        if (i + 1 + suffix.size() > runes.size()) continue;
        bool equal = true;
        for (size_t j = 0; j < suffix.size(); ++j) {
          uint32_t c = runes[i + 1 + j].codepoint;
          if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
          if (c != static_cast<uint32_t>(suffix[j])) equal = false;
        }
        if (equal) { emit(i, i + 1 + suffix.size()); i += 1 + suffix.size(); matched = true; break; }
      }
      if (matched) continue;
    }

    // Optional single prefix character followed by letters/combining marks.
    size_t start = i;
    size_t letters = i;
    if (!IsNewline(runes[i].codepoint) && !IsLetter(runes[i].codepoint) &&
        !IsNumber(runes[i].codepoint)) {
      letters = i + 1;
    }
    if (letters < runes.size() && (IsLetter(runes[letters].codepoint) || IsMark(runes[letters].codepoint))) {
      size_t end = letters + 1;
      while (end < runes.size() && (IsLetter(runes[end].codepoint) || IsMark(runes[end].codepoint))) ++end;
      emit(start, end); i = end; continue;
    }

    if (IsNumber(runes[i].codepoint)) { emit(i, i + 1); ++i; continue; }

    // Optional ASCII space, punctuation/symbol run, then line breaks.
    size_t symbols = i + (runes[i].codepoint == ' ' ? 1 : 0);
    if (symbols < runes.size() && !IsSpace(runes[symbols].codepoint) && !IsWord(runes[symbols].codepoint)) {
      size_t end = symbols + 1;
      while (end < runes.size() && !IsSpace(runes[end].codepoint) && !IsWord(runes[end].codepoint)) ++end;
      while (end < runes.size() && IsNewline(runes[end].codepoint)) ++end;
      emit(i, end); i = end; continue;
    }

    // Whitespace/newline alternatives. Preserve one leading space for the next word.
    if (IsSpace(runes[i].codepoint)) {
      size_t end = i + 1;
      while (end < runes.size() && IsSpace(runes[end].codepoint)) ++end;
      if (!IsNewline(runes[i].codepoint) && end < runes.size() && end - i > 1) --end;
      emit(i, end); i = end; continue;
    }
    emit(i, i + 1); ++i;
  }
  return pieces;
}

std::vector<std::string> QwenTokenizer::ByteEncode(const std::string& text) const {
  std::vector<std::string> result;
  result.reserve(text.size());
  for (unsigned char byte : text) result.push_back(byte_to_symbol_[byte]);
  return result;
}

std::vector<std::string> QwenTokenizer::Bpe(const std::string& piece) const {
  auto symbols = ByteEncode(piece);
  while (symbols.size() > 1) {
    size_t best_rank = std::numeric_limits<size_t>::max();
    size_t best_index = symbols.size();
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const auto found = merge_rank_.find({symbols[i], symbols[i + 1]});
      if (found != merge_rank_.end() && found->second < best_rank) {
        best_rank = found->second;
        best_index = i;
      }
    }
    if (best_index == symbols.size()) break;
    symbols[best_index] += symbols[best_index + 1];
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
  }
  return symbols;
}

std::vector<int64_t> QwenTokenizer::Encode(const std::string& text) const {
  std::vector<int64_t> ids;
  size_t position = 0;
  while (position < text.size()) {
    const std::string* special = nullptr;
    for (const auto& candidate : specials_by_length_) {
      if (text.compare(position, candidate.size(), candidate) == 0) { special = &candidate; break; }
    }
    if (special) {
      ids.push_back(special_to_id_.at(*special));
      position += special->size();
      continue;
    }
    size_t next = text.size();
    for (const auto& candidate : specials_by_length_) {
      const size_t found = text.find(candidate, position);
      if (found != std::string::npos) next = std::min(next, found);
    }
    const std::string ordinary = text.substr(position, next - position);
    for (const auto& piece : Pretokenize(ordinary)) {
      for (const auto& token : Bpe(piece)) {
        const auto found = vocab_.find(token);
        if (found == vocab_.end()) throw std::runtime_error("tokenizer produced an unknown BPE token");
        ids.push_back(found->second);
      }
    }
    position = next;
  }
  return ids;
}

std::string QwenTokenizer::DecodeToken(int64_t id, bool skip_special) const {
  if (id_to_special_.count(id))
    return skip_special && skip_special_ids_.count(id) ? std::string{} : id_to_special_.at(id);
  if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return {};
  const std::string& token = id_to_token_[static_cast<size_t>(id)];
  std::string result;
  for (const auto& rune : DecodeUtf8(token)) {
    const std::string symbol = token.substr(rune.begin, rune.end - rune.begin);
    const auto found = symbol_to_byte_.find(symbol);
    if (found == symbol_to_byte_.end()) throw std::runtime_error("invalid byte-level token");
    result.push_back(static_cast<char>(found->second));
  }
  return result;
}
