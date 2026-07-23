#!/usr/bin/env bash
set -euo pipefail

# Run GEPA and MIPROv2 baselines on MGSM and HotPotQA — all 4 in parallel.
# Assumes 8 vLLM servers on ports 8000–8007 serving qwen3-local.
#
# Server allocation (2 per experiment):
#   GEPA-MGSM       → ports 8000-8001
#   GEPA-HotPotQA   → ports 8002-8003
#   MIPROv2-MGSM    → ports 8004-8005
#   MIPROv2-HotPotQA→ ports 8006-8007
#
# Usage:
#   bash scripts/run_baselines.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Activate virtualenv
VENV_DIR="$PROJECT_DIR/.venv"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
  echo "Activated virtualenv: $VENV_DIR"
else
  echo "WARNING: No virtualenv found at $VENV_DIR — using system python"
fi

# Scripts use relative paths (datasets/, results/), so run from project root
cd "$PROJECT_DIR"

MODEL="${MODEL:-qwen3-local}"

echo "============================================"
echo " DSPy Baselines: GEPA + MIPROv2"
echo " Tasks: MGSM, HotPotQA"
echo " Model: $MODEL  (8 servers, ports 8000-8007)"
echo " All 4 experiments in parallel (2 servers each)"
echo " Logs:  $LOG_DIR"
echo "============================================"
echo ""

PIDS=()
NAMES=()

# ── 1. GEPA MGSM (ports 8000-8001) ──────────────────────────────────
python3 "$SCRIPT_DIR/baseline_dspy_gepa_mgsm.py" \
  --model "$MODEL" \
  --base-urls http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 \
  --num-threads 24 \
  > "$LOG_DIR/gepa_mgsm.log" 2>&1 &
PIDS+=($!)
NAMES+=("GEPA-MGSM")
echo "  Started GEPA-MGSM         PID=${PIDS[-1]}  ports 8000-8001  → gepa_mgsm.log"

# ── 2. GEPA HotPotQA (ports 8002-8003) ──────────────────────────────
python3 "$SCRIPT_DIR/baseline_dspy_gepa_hotpotqa.py" \
  --model "$MODEL" \
  --base-urls http://127.0.0.1:8002/v1 http://127.0.0.1:8003/v1 \
  --num-threads 24 \
  > "$LOG_DIR/gepa_hotpotqa.log" 2>&1 &
PIDS+=($!)
NAMES+=("GEPA-HotPotQA")
echo "  Started GEPA-HotPotQA     PID=${PIDS[-1]}  ports 8002-8003  → gepa_hotpotqa.log"

# ── 3. MIPROv2 MGSM (ports 8004-8005) ───────────────────────────────
python3 "$SCRIPT_DIR/baseline_dspy_mipro_mgsm.py" \
  --model "$MODEL" \
  --base-urls http://127.0.0.1:8004/v1 http://127.0.0.1:8005/v1 \
  --num-threads 24 \
  > "$LOG_DIR/mipro_mgsm.log" 2>&1 &
PIDS+=($!)
NAMES+=("MIPROv2-MGSM")
echo "  Started MIPROv2-MGSM      PID=${PIDS[-1]}  ports 8004-8005  → mipro_mgsm.log"

# ── 4. MIPROv2 HotPotQA (ports 8006-8007) ───────────────────────────
python3 "$SCRIPT_DIR/baseline_dspy_mipro_hotpotqa.py" \
  --model "$MODEL" \
  --base-urls http://127.0.0.1:8006/v1 http://127.0.0.1:8007/v1 \
  --num-threads 24 \
  > "$LOG_DIR/mipro_hotpotqa.log" 2>&1 &
PIDS+=($!)
NAMES+=("MIPROv2-HotPotQA")
echo "  Started MIPROv2-HotPotQA  PID=${PIDS[-1]}  ports 8006-8007  → mipro_hotpotqa.log"

echo ""
echo "Waiting for all 4 experiments..."
echo ""

FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  ✓ ${NAMES[$i]} finished"
  else
    echo "  ✗ ${NAMES[$i]} failed (exit $?)"
    FAIL=1
  fi
done

echo ""
echo "============================================"
if [ $FAIL -eq 0 ]; then
  echo " All 4 experiments completed successfully."
else
  echo " Some experiments failed — check logs."
fi
echo " Logs:"
echo "   $LOG_DIR/gepa_mgsm.log"
echo "   $LOG_DIR/gepa_hotpotqa.log"
echo "   $LOG_DIR/mipro_mgsm.log"
echo "   $LOG_DIR/mipro_hotpotqa.log"
echo "============================================"
