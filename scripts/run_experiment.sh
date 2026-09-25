#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG_PATH="${1:-configs/experiment.yaml}"
mkdir -p logs results
exec python src/main.py --config "$CONFIG_PATH"
