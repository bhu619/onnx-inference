#!/usr/bin/env bash

# Initialize the Git submodules under third-party/:
#   - ONNX Runtime v1.27.1
#   - ONNX Runtime GenAI v0.14.0
#
# The superproject records a pinned commit for each submodule, so this script
# does not move them to the latest commit of any remote branch. --recursive
# also initializes the nested submodules required by ONNX Runtime.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SUBMODULES=(
  "third-party/onnxruntime"
  "third-party/onnxruntime-genai"
)

if ! git -C "${PROJECT_ROOT}" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "Error: ${PROJECT_ROOT} is not a Git work tree." >&2
  exit 1
fi

echo "Syncing submodule URLs..."
git -C "${PROJECT_ROOT}" submodule sync --recursive -- "${SUBMODULES[@]}"

echo "Initializing third-party submodules and their nested dependencies..."
git -C "${PROJECT_ROOT}" submodule update --init --recursive -- "${SUBMODULES[@]}"

echo "Submodule initialization complete:"
git -C "${PROJECT_ROOT}" submodule status --recursive -- "${SUBMODULES[@]}"
