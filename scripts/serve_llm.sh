#!/usr/bin/env bash
set -euo pipefail

# Usage: MODEL=qwen3 ./serve_llm.sh   (default: local_qwen3_model / qwen3-local)
#        MODEL=qwen3.5 ./serve_llm.sh
#        MODEL=qwen3-4b-2507 ./serve_llm.sh
#        MODEL=ministral-14b ./serve_llm.sh
MODEL="${MODEL:-qwen3}"
HOST="${SERVE_HOST:-0.0.0.0}"
PORT="${SERVE_PORT:-${PORT:-8000}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

if [[ "$MODEL" == "ministral-14b" ]]; then
  MODEL_PATH="${MODEL_PATH:-$(dirname "$0")/../local_ministral_model}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ministral-14b-local}"
  TOOL_CALL_PARSER="mistral"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  TP_SIZE="${TP_SIZE:-1}"
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
elif [[ "$MODEL" == "qwen3.5" ]]; then
  MODEL_PATH="${MODEL_PATH:-$(dirname "$0")/../local_qwen35_model}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-local}"
  TOOL_CALL_PARSER="qwen3_coder"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  TP_SIZE="${TP_SIZE:-1}"
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
elif [[ "$MODEL" == "qwen3-4b-2507" ]]; then
  MODEL_PATH="${MODEL_PATH:-$(dirname "$0")/../local_qwen3_model}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-local}"
  TOOL_CALL_PARSER="hermes"
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
else
  MODEL_PATH="${MODEL_PATH:-$(dirname "$0")/../local_qwen3_model}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-local}"
  TOOL_CALL_PARSER="qwen3_coder"
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
fi

echo "Serving model: $MODEL (path=$MODEL_PATH, parser=$TOOL_CALL_PARSER)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dtype "auto" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER"
