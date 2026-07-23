#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# run_ablation_parallel.sh — Run all 4 ablation groups in parallel
#
# Prerequisites:
#   8 vLLM servers on ports 8000-8007 serving qwen3.5-local:
#     NUM_SERVERS=8 MODEL=qwen3.5 ./scripts/serve_all.sh
#
# Usage:
#   ./scripts/run_ablation_parallel.sh
#   ./scripts/run_ablation_parallel.sh --resume logs/ablation_20260423_204500
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RESUME_FLAG=""
RESUME_DIR=""
if [[ "${1:-}" == "--resume" && -n "${2:-}" ]]; then
    RESUME_DIR="$2"
    RESUME_FLAG="--resume $RESUME_DIR"
    LOG_DIR="$RESUME_DIR"
    echo "Resuming into: ${LOG_DIR}"
else
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    LOG_DIR="./logs/ablation_${TIMESTAMP}"
fi
mkdir -p "$LOG_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Parallel Ablation — 4 groups across 8 GPUs                ║"
echo "║  Group 1: epsilon × MMLU   (ports 8000-8001)               ║"
echo "║  Group 2: epsilon × IFEval (ports 8002-8003)               ║"
echo "║  Group 3: abgate  × MMLU   (ports 8004-8005)               ║"
echo "║  Group 4: abgate  × IFEval (ports 8006-8007)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Logs: ${LOG_DIR}/"
echo ""

# Launch 4 runners in parallel, each handling one study×benchmark group
bash scripts/run_ablation.sh epsilon_mmlu  ${RESUME_FLAG} > "${LOG_DIR}/runner_epsilon_mmlu.out"  2>&1 &
PID1=$!
bash scripts/run_ablation.sh epsilon_ifeval ${RESUME_FLAG} > "${LOG_DIR}/runner_epsilon_ifeval.out" 2>&1 &
PID2=$!
bash scripts/run_ablation.sh abgate_mmlu   ${RESUME_FLAG} > "${LOG_DIR}/runner_abgate_mmlu.out"   2>&1 &
PID3=$!
bash scripts/run_ablation.sh abgate_ifeval ${RESUME_FLAG} > "${LOG_DIR}/runner_abgate_ifeval.out"  2>&1 &
PID4=$!

echo "Runner PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Waiting for all groups to finish..."
echo ""

# Wait and report as each finishes
FAIL=0
for PID in $PID1 $PID2 $PID3 $PID4; do
    if wait "$PID"; then
        echo "  ✓ PID $PID finished"
    else
        echo "  ✗ PID $PID exited with error"
        FAIL=1
    fi
done

echo ""
echo "━━━ All parallel groups complete ━━━"

# Merge all per-group summaries into one
echo "label,config,final_accuracy" > "${LOG_DIR}/summary.csv"
for f in "${LOG_DIR}"/runner_*.out; do
    # The sub-runners also write to the shared LOG_DIR, so just collect from logs
    true
done

# Rebuild summary from all logs
for f in "${LOG_DIR}"/*.log; do
    [[ -f "$f" ]] || continue
    label=$(basename "$f" .log)
    acc=$(grep -oP 'LastEvalAcc=\K[0-9.]+' "$f" 2>/dev/null | sort -rn | head -1 || echo "N/A")
    if [[ -z "$acc" ]]; then acc="N/A"; fi
    echo "${label},,${acc}" >> "${LOG_DIR}/summary.csv"
done

echo ""
echo "Summary: ${LOG_DIR}/summary.csv"
cat "${LOG_DIR}/summary.csv"

exit $FAIL
