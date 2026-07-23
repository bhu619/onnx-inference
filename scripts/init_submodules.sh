#!/usr/bin/env bash

# Initialize the Git submodules under third-party/:
#   - ONNX Runtime v1.27.1
#   - ONNX Runtime GenAI v0.14.0
#
# The superproject records a pinned commit for each submodule, so this script
# does not move them to the latest commit of any remote branch. --recursive
# also initializes the nested submodules required by ONNX Runtime.
#
# All submodules are initialized by default; pass one or more submodule names
# to initialize only those. Use --dry-run to print the git commands without
# executing them. See --help for details.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ALL_SUBMODULES=(
  "third-party/onnxruntime"
  "third-party/onnxruntime-genai"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [--dry-run] [-h|--help] [SUBMODULE ...]

Initialize the Git submodules under third-party/ (default: all).

Submodules:
  onnxruntime          third-party/onnxruntime (ONNX Runtime v1.27.1)
  onnxruntime-genai    third-party/onnxruntime-genai (ONNX Runtime GenAI v0.14.0)

Options:
  --dry-run            Print the git commands without executing them
  -h, --help           Show this help and exit
EOF
}

DRY_RUN=0
SELECTED=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    onnxruntime|third-party/onnxruntime)
      SELECTED+=("third-party/onnxruntime")
      ;;
    onnxruntime-genai|third-party/onnxruntime-genai)
      SELECTED+=("third-party/onnxruntime-genai")
      ;;
    *)
      echo "Error: unknown argument '$1'." >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("${ALL_SUBMODULES[@]}")
fi

if ! git -C "${PROJECT_ROOT}" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "Error: ${PROJECT_ROOT} is not a Git work tree." >&2
  exit 1
fi

# run executes the given command, or only prints it in dry-run mode.
run() {
  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "Dry run: printing commands without executing them."
fi

echo "Syncing submodule URLs..."
run git -C "${PROJECT_ROOT}" submodule sync --recursive -- "${SELECTED[@]}"

echo "Initializing submodules: ${SELECTED[*]} ..."
run git -C "${PROJECT_ROOT}" submodule update --init --recursive -- "${SELECTED[@]}"

echo "Submodule status:"
run git -C "${PROJECT_ROOT}" submodule status --recursive -- "${SELECTED[@]}"
