#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama CLI not found."
  echo "Install it first, then re-run this script."
  exit 1
fi

if [ ! -d "venv" ]; then
  echo "Expected virtualenv at ./venv not found."
  exit 1
fi

set -a
source .env
set +a

MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
MODELS_DIR="${OLLAMA_MODELS:-$HOME/.ollama/models}"

echo "Pulling model: $MODEL"
echo "Model directory: $MODELS_DIR"
ollama pull "$MODEL"

echo "Running smoke test..."
./venv/bin/python scripts/check_ollama.py
