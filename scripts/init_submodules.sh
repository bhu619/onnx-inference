#!/usr/bin/env bash

# Initialize the Git submodules under third-party/:
#   - ONNX Runtime v1.27.1
#   - ONNX Runtime GenAI v0.14.0
#
# The superproject records a pinned commit for each submodule, so this script
# does not move them to the latest commit of any remote branch. --recursive
# also initializes the nested submodules required by ONNX Runtime.
#
# Use --all to initialize all submodules, or --submodule (-s) to select specific
# ones. Use --dry-run to print the git commands without executing them.
# See --help for details.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ALL_SUBMODULES=(
  "third-party/onnxruntime"
  "third-party/onnxruntime-genai"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Initialize the Git submodules under third-party/.
When called without arguments, shows this help and exits.

Submodules:
  onnxruntime          third-party/onnxruntime (ONNX Runtime v1.27.1)
  onnxruntime-genai    third-party/onnxruntime-genai (ONNX Runtime GenAI v0.14.0)

Options:
  --all                Initialize all submodules
  -s, --submodule NAME Add a submodule (repeatable, comma-separated accepted)
  --dry-run            Print the git commands without executing them
  -h, --help           Show this help and exit

Examples:
  $(basename "$0") --all
  $(basename "$0") --all --dry-run
  $(basename "$0") --submodule onnxruntime
  $(basename "$0") --submodule onnxruntime --submodule onnxruntime-genai
  $(basename "$0") --submodule onnxruntime, onnxruntime-genai
  $(basename "$0") --submodule=onnxruntime
  $(basename "$0") -s onnxruntime
  $(basename "$0") -s onnxruntime,onnxruntime-genai
  $(basename "$0") -s onnxruntime --dry-run
EOF
}

DRY_RUN=0
SELECTED=()
HAS_ALL=0

# resolve_submodule adds a normalized submodule path to SELECTED.
# Accepts short names (onnxruntime, onnxruntime-genai) or full paths.
resolve_submodule() {
  local name="$1"
  case "$name" in
    onnxruntime|third-party/onnxruntime)
      SELECTED+=("third-party/onnxruntime") ;;
    onnxruntime-genai|third-party/onnxruntime-genai)
      SELECTED+=("third-party/onnxruntime-genai") ;;
    *)
      echo "Error: unknown submodule '$name'." >&2
      usage >&2
      exit 1 ;;
  esac
}

# parse_submodules splits a comma-separated list and resolves each item.
parse_submodules() {
  local IFS=','
  for item in $1; do
    # Trim leading/trailing whitespace.
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    [[ -n "$item" ]] && resolve_submodule "$item"
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --all)
      HAS_ALL=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -s|--submodule)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --submodule requires a value." >&2
        usage >&2
        exit 1
      fi
      parse_submodules "$2"
      shift
      ;;
    --submodule=*)
      parse_submodules "${1#*=}"
      ;;
    *)
      # Treat bare word (not starting with -) as a submodule name.
      if [[ "$1" != -* ]]; then
        resolve_submodule "$1"
      else
        echo "Error: unknown argument '$1'." >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
  shift
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  if [[ ${HAS_ALL} -eq 1 ]]; then
    SELECTED=("${ALL_SUBMODULES[@]}")
  else
    usage
    exit 0
  fi
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

if [[ ${DRY_RUN} -eq 0 ]]; then
  echo "Syncing submodule URLs..."
fi
run git -C "${PROJECT_ROOT}" submodule sync --recursive -- "${SELECTED[@]}"

if [[ ${DRY_RUN} -eq 0 ]]; then
  echo "Initializing submodules: ${SELECTED[*]} ..."
fi
run git -C "${PROJECT_ROOT}" submodule update --init --recursive -- "${SELECTED[@]}"

if [[ ${DRY_RUN} -eq 0 ]]; then
  echo "Submodule status:"
fi
run git -C "${PROJECT_ROOT}" submodule status --recursive -- "${SELECTED[@]}"
