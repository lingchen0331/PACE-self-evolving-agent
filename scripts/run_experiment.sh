#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiment.yaml}"

python src/main.py --config "$CONFIG_PATH"
