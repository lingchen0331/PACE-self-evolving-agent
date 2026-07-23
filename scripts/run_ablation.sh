#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# run_ablation.sh — Sequential ablation study runner
#
# Usage:
#   ./scripts/run_ablation.sh [study] [--resume <dir>]
#
# study: epsilon | abgate | epsilon_mmlu | epsilon_ifeval
#        | abgate_mmlu | abgate_ifeval | all
#
# Examples:
#   ./scripts/run_ablation.sh all
#   ./scripts/run_ablation.sh epsilon_mmlu --resume logs/ablation_20260423_204500
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

STUDY="${1:-all}"
RESUME_DIR=""

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -n "$RESUME_DIR" && -d "$RESUME_DIR" ]]; then
    LOG_DIR="$RESUME_DIR"
    echo "Resuming into: ${LOG_DIR}"
else
    TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
    LOG_DIR="./logs/ablation_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"
fi

if [[ ! -f "${LOG_DIR}/summary.csv" ]]; then
    echo "label,config,final_accuracy" > "${LOG_DIR}/summary.csv"
fi

run_config() {
    local config="$1"
    local label="$2"
    local logfile="${LOG_DIR}/${label}.log"

    if [[ -f "$logfile" ]] && grep -q 'LastEvalAcc=' "$logfile" && grep -q 'Outer Loop Stop\|Prompt Saturation Reached' "$logfile"; then
        local acc
        acc=$(grep -oP 'LastEvalAcc=\K[0-9.]+' "$logfile" | sort -rn | head -1 || echo "N/A")
        echo "━━━ Skipping: ${label} (already completed, best acc=${acc}) ━━━"
        if ! grep -q "^${label}," "${LOG_DIR}/summary.csv"; then
            echo "${label},${config},${acc}" >> "${LOG_DIR}/summary.csv"
        fi
        return
    fi

    echo "━━━ Running: ${label} (config: ${config}) ━━━"
    echo "    Log: ${logfile}"
    python src/main.py --config "$config" > "$logfile" 2>&1 || true
    local acc
    acc=$(grep -oP 'LastEvalAcc=\K[0-9.]+' "$logfile" | sort -rn | head -1 || echo "N/A")
    echo "    Best accuracy: ${acc}"
    grep -v "^${label}," "${LOG_DIR}/summary.csv" > "${LOG_DIR}/summary.csv.tmp" || true
    mv "${LOG_DIR}/summary.csv.tmp" "${LOG_DIR}/summary.csv"
    echo "${label},${config},${acc}" >> "${LOG_DIR}/summary.csv"
}

# ── Epsilon × MMLU ───────────────────────────────────────────────────
if [[ "$STUDY" == "epsilon" || "$STUDY" == "epsilon_mmlu" || "$STUDY" == "all" ]]; then
    echo "── Epsilon × MMLU (ports 8000-8001) ──"
    for eps in eps0 eps005 eps01 eps02 eps05 eps100; do
        run_config "configs/ablation_epsilon_mmlu_${eps}.yaml" "epsilon_mmlu_${eps}"
    done
fi

# ── Epsilon × IFEval ─────────────────────────────────────────────────
if [[ "$STUDY" == "epsilon" || "$STUDY" == "epsilon_ifeval" || "$STUDY" == "all" ]]; then
    echo "── Epsilon × IFEval (ports 8002-8003) ──"
    for eps in eps0 eps005 eps01 eps02 eps05 eps100; do
        run_config "configs/ablation_epsilon_ifeval_${eps}.yaml" "epsilon_ifeval_${eps}"
    done
fi

# ── A/B Gate × MMLU ──────────────────────────────────────────────────
if [[ "$STUDY" == "abgate" || "$STUDY" == "abgate_mmlu" || "$STUDY" == "all" ]]; then
    echo "── A/B Gate × MMLU (ports 8004-8005) ──"
    for variant in off zero default strict; do
        run_config "configs/ablation_abgate_mmlu_${variant}.yaml" "abgate_mmlu_${variant}"
    done
fi

# ── A/B Gate × IFEval ────────────────────────────────────────────────
if [[ "$STUDY" == "abgate" || "$STUDY" == "abgate_ifeval" || "$STUDY" == "all" ]]; then
    echo "── A/B Gate × IFEval (ports 8006-8007) ──"
    for variant in off zero default strict; do
        run_config "configs/ablation_abgate_ifeval_${variant}.yaml" "abgate_ifeval_${variant}"
    done
fi

echo ""
echo "━━━ Group ${STUDY} complete ━━━"
echo "Summary: ${LOG_DIR}/summary.csv"
cat "${LOG_DIR}/summary.csv"
