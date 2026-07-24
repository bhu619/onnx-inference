#pragma once

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace terminal_ui {

inline constexpr const char* kGray = "\033[90m";
inline constexpr const char* kGreen = "\033[32m";
inline constexpr const char* kWhite = "\033[97m";
inline constexpr const char* kPurple = "\033[35m";
inline constexpr const char* kReset = "\033[0m";
inline constexpr const char* kThinkStart = "<think>";
inline constexpr const char* kThinkEnd = "</think>";

inline volatile std::sig_atomic_t g_generating = 0;
inline volatile std::sig_atomic_t g_interrupt_requested = 0;

inline void HandleInterrupt(int) {
  if (g_generating) {
    g_interrupt_requested = 1;
  } else {
    static constexpr char kResetTerminal[] = "\033[0m";
    const ssize_t ignored =
        ::write(STDOUT_FILENO, kResetTerminal, sizeof(kResetTerminal) - 1);
    static_cast<void>(ignored);
    std::_Exit(130);
  }
}

inline void InstallInterruptHandler() {
  std::signal(SIGINT, HandleInterrupt);
}

class GenerationGuard {
 public:
  GenerationGuard() {
    g_interrupt_requested = 0;
    g_generating = 1;
  }

  ~GenerationGuard() {
    Finish();
  }

  GenerationGuard(const GenerationGuard&) = delete;
  GenerationGuard& operator=(const GenerationGuard&) = delete;

  bool Interrupted() const {
    return g_interrupt_requested != 0;
  }

  void Finish() {
    if (!active_) return;
    g_generating = 0;
    g_interrupt_requested = 0;
    active_ = false;
  }

 private:
  bool active_ = true;
};

inline std::filesystem::path ModelCachePath(const std::filesystem::path& relative) {
  const char* home = std::getenv("HOME");
  if (home == nullptr || *home == '\0') {
    throw std::runtime_error("HOME is not set; specify the model directory with --model");
  }
  return std::filesystem::path(home) / ".cache/models" / relative;
}

inline void PrintLoadingMessage() {
  std::cout << "\n\nLoading tokenizer and ONNX sessions..." << std::flush;
}

inline void PrintChatHeader(const std::filesystem::path& model) {
  std::cout << "\n\nmodel: "
            << std::filesystem::absolute(model).lexically_normal().string()
            << "\n\navailable commands:\n"
            << "  /exit or Ctrl+C     stop or exit\n\n";
}

inline bool ReadPrompt(std::string& prompt) {
  if (prompt == "/exit") return false;
  while (prompt.empty()) {
    std::cout << kGreen << "> " << std::flush;
    if (!std::getline(std::cin, prompt)) {
      std::cout << kReset;
      return false;
    }
    std::cout << kReset;
    if (prompt == "/exit") return false;
  }
  return true;
}

// Streams model text while replacing Qwen's thinking tags with colored,
// human-readable section markers.
class TerminalOutput {
 public:
  explicit TerminalOutput(bool thinking) : thinking_(thinking) {
    std::cout << '\n';
    if (thinking_) {
      std::cout << kGray << "[Start thinking]\n\n";
    } else {
      std::cout << kWhite;
    }
  }

  void Write(const std::string& text) {
    pending_ += text;
    TrimLeadingNewlines();
    if (pending_.empty()) return;
    while (!pending_.empty()) {
      const size_t start = pending_.find(kThinkStart);
      const size_t end = pending_.find(kThinkEnd);
      size_t marker = std::string::npos;
      bool starts_thinking = false;
      if (start != std::string::npos && (end == std::string::npos || start < end)) {
        marker = start;
        starts_thinking = true;
      } else if (end != std::string::npos) {
        marker = end;
      }

      if (marker != std::string::npos) {
        std::string content = pending_.substr(0, marker);
        if (!starts_thinking) {
          while (!content.empty() && (content.back() == '\n' || content.back() == '\r')) {
            content.pop_back();
          }
        }
        std::cout << content;
        pending_.erase(0, marker + (starts_thinking ? 7 : 8));
        trim_leading_newlines_ = true;
        while (!pending_.empty() && (pending_.front() == '\n' || pending_.front() == '\r')) {
          pending_.erase(0, 1);
        }
        if (!pending_.empty()) trim_leading_newlines_ = false;
        if (starts_thinking) {
          if (!thinking_) std::cout << kGray << "[Start thinking]\n\n";
          thinking_ = true;
        } else if (thinking_) {
          std::cout << "\n[End thinking]\n\n" << kWhite;
          thinking_ = false;
        }
        continue;
      }

      const size_t safe = SafePrefixLength();
      if (safe == 0) break;
      std::cout << pending_.substr(0, safe) << std::flush;
      pending_.erase(0, safe);
    }
  }

  void Finish() {
    if (thinking_) {
      while (!pending_.empty() && (pending_.back() == '\n' || pending_.back() == '\r')) {
        pending_.pop_back();
      }
    }
    std::cout << pending_;
    pending_.clear();
    if (thinking_) {
      std::cout << "\n[End thinking]\n";
      thinking_ = false;
    }
    std::cout << kReset;
  }

 private:
  size_t SafePrefixLength() const {
    size_t keep = 0;
    for (const std::string marker : {std::string(kThinkStart), std::string(kThinkEnd)}) {
      const size_t limit = std::min(pending_.size(), marker.size() - 1);
      for (size_t length = 1; length <= limit; ++length) {
        if (pending_.compare(pending_.size() - length, length, marker, 0, length) == 0) {
          keep = std::max(keep, length);
        }
      }
    }
    if (thinking_) {
      size_t newlines = 0;
      while (newlines < pending_.size()) {
        const char c = pending_[pending_.size() - newlines - 1];
        if (c != '\n' && c != '\r') break;
        ++newlines;
      }
      keep = std::max(keep, newlines);
    }
    return pending_.size() - keep;
  }

  void TrimLeadingNewlines() {
    if (!trim_leading_newlines_) return;
    while (!pending_.empty() && (pending_.front() == '\n' || pending_.front() == '\r')) {
      pending_.erase(0, 1);
    }
    if (!pending_.empty()) trim_leading_newlines_ = false;
  }

  bool thinking_;
  bool trim_leading_newlines_ = false;
  std::string pending_;
};

inline void PrintStats(size_t prompt_tokens, size_t generated_tokens,
                       double first_token_seconds, double prompt_rate,
                       double generation_rate) {
  std::cout << kPurple << "\n\n[ Prompt: " << prompt_tokens << " tokens, "
            << std::fixed << std::setprecision(1) << prompt_rate
            << " t/s | Generation: " << generated_tokens << " tokens, "
            << generation_rate << " t/s | First token: " << std::setprecision(3)
            << first_token_seconds << " s ]" << kReset << "\n\n";
}

}  // namespace terminal_ui
