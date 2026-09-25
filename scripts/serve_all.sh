#!/usr/bin/env bash
set -euo pipefail
# Launch one server per GPU; existing GPU workloads are never terminated.
# Example: MODEL=qwen3.5 NUM_SERVERS=2 bash scripts/serve_all.sh
MODEL="${MODEL:-qwen3}"
NUM_SERVERS="${NUM_SERVERS:-2}"
BASE_PORT="${BASE_PORT:-8000}"
READY_ATTEMPTS="${READY_ATTEMPTS:-120}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"

for value in "$NUM_SERVERS" "$BASE_PORT" "$READY_ATTEMPTS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SERVERS, BASE_PORT, and READY_ATTEMPTS must be positive integers." >&2
    exit 2
  fi
done
# Check every device before starting any server.
for ((i=0; i<NUM_SERVERS; i++)); do
  occupied=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -n "$occupied" ]]; then
    echo "GPU $i already has compute processes. Choose idle GPUs before launching." >&2
    exit 1
  fi
done
# Refuse occupied ports so an unrelated server cannot satisfy our health check.
python3 - "$BASE_PORT" "$NUM_SERVERS" <<'PY'
import socket
import sys
for port in range(int(sys.argv[1]), int(sys.argv[1]) + int(sys.argv[2])):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(f"Port {port} is unavailable: {exc}")
PY
mkdir -p "$LOG_DIR"
PIDS=()
for ((i=0; i<NUM_SERVERS; i++)); do
  port=$((BASE_PORT + i))
  logfile="$LOG_DIR/vllm_${MODEL}_${port}.log"
  MODEL="$MODEL" SERVE_PORT="$port" CUDA_VISIBLE_DEVICES="$i" \
    nohup bash "$SCRIPT_DIR/serve_llm.sh" > "$logfile" 2>&1 &
  PIDS+=("$!")
  echo "GPU $i, port $port: PID $! -> $logfile"
done
echo "To stop these servers: kill ${PIDS[*]}"
failed=0
for ((i=0; i<NUM_SERVERS; i++)); do
  port=$((BASE_PORT + i))
  ready=0
  for ((attempt=0; attempt<READY_ATTEMPTS; attempt++)); do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      echo "Server on port $port exited; inspect its log." >&2
      break
    fi
    if curl --max-time 2 -sf "http://127.0.0.1:$port/health" >/dev/null; then
      echo "Port $port ready"
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "Server on port $port failed to become ready." >&2
    failed=1
  fi
done
exit "$failed"
