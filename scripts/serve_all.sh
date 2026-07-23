#!/usr/bin/env bash
set -euo pipefail

# Launches N vLLM servers for the same model on separate GPUs.
# Usage: ./scripts/serve_all.sh                        (2 servers, ministral-14b)
#        MODEL=qwen3.5 NUM_SERVERS=4 ./scripts/serve_all.sh
#        NUM_SERVERS=4 ./scripts/serve_all.sh           (4 servers on GPUs 0-3)

MODEL="${MODEL:-ministral-14b}"
NUM_SERVERS="${NUM_SERVERS:-2}"
BASE_PORT="${BASE_PORT:-8000}"
SCRIPT_DIR="$(dirname "$0")"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

echo "Launching ${NUM_SERVERS} × ${MODEL} servers (ports ${BASE_PORT}–$((BASE_PORT + NUM_SERVERS - 1)))..."

# --- Free up target GPUs if occupied ---
echo "Checking GPUs 0–$((NUM_SERVERS - 1)) for existing processes..."
for GPU_ID in $(seq 0 $((NUM_SERVERS - 1))); do
  PIDS=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader,nounits \
    | grep "$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits \
              | awk -F', ' -v idx="$GPU_ID" '$1==idx {print $2}')" \
    | awk -F', ' '{print $1}' || true)
  if [[ -n "$PIDS" ]]; then
    echo "  GPU $GPU_ID occupied by PID(s): $PIDS — killing..."
    for P in $PIDS; do
      kill "$P" 2>/dev/null || true
    done
    sleep 2
    for P in $PIDS; do
      kill -9 "$P" 2>/dev/null || true
    done
  else
    echo "  GPU $GPU_ID is free"
  fi
done
sleep 1

# --- Launch servers ---
PIDS=()
for i in $(seq 0 $((NUM_SERVERS - 1))); do
  PORT=$((BASE_PORT + i))
  LOGFILE="${LOG_DIR}/vllm_${MODEL}_${PORT}.log"
  echo "  Starting server $((i + 1))/${NUM_SERVERS}: GPU $i, port $PORT"
  MODEL="$MODEL" SERVE_PORT="$PORT" CUDA_VISIBLE_DEVICES="$i" \
    nohup "$SCRIPT_DIR/serve_llm.sh" > "$LOGFILE" 2>&1 &
  PIDS+=($!)
  echo "    PID=${PIDS[-1]} → $(basename "$LOGFILE")"
done

echo ""
echo "Waiting for all servers to be ready..."
for i in $(seq 0 $((NUM_SERVERS - 1))); do
  PORT=$((BASE_PORT + i))
  for attempt in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
      echo "  ✓ Port ${PORT} ready"
      break
    fi
    if [[ $attempt -eq 120 ]]; then
      echo "  ✗ Port ${PORT} timed out after 10 minutes"
    fi
    sleep 5
  done
done

echo ""
echo "All ${NUM_SERVERS} servers launched."
echo "To stop: kill ${PIDS[*]}"
