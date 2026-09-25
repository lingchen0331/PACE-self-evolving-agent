#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
uv venv --python "${PYTHON_VERSION:-3.12}" .venv-serving
uv pip install --python .venv-serving/bin/python -r requirements-serving.txt
echo "Serving environment ready: source .venv-serving/bin/activate"
echo "Download a model explicitly: python model_downloader.py --target qwen3"
