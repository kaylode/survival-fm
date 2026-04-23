#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_cr.sh — Competing-risks survival benchmark runner
#
# Usage
# ─────
#   ./run_cr.sh                            # all CR models, all CR datasets
#   ./run_cr.sh classical_cr               # Aalen-Johansen, Fine-Gray, Cox CR
#   ./run_cr.sh deep_cr                    # DeepHit CR
#   ./run_cr.sh fm_embedding               # all FM frozen-embedding CR models
#   ./run_cr.sh fm_joint                   # all FM jointly-trained CR models
#   ./run_cr.sh fm_zeroshot                # all FM zero-shot CR models
#   ./run_cr.sh tabpfn_cr                  # all TabPFN CR variants
#   ./run_cr.sh tabdpt_cr                  # all TabDPT CR variants
#   ./run_cr.sh tabicl_cr                  # all TabICL CR variants
#   ./run_cr.sh all FRAMINGHAM             # all CR models, one dataset
#
# Dataset keywords (2nd positional arg)
# ──────────────────────────────────────
#   (any single dataset name from BENCHMARK_DATASETS)
#   (default) → FRAMINGHAM PBC2 SUPPORT_CR SYNTHETIC_CR
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON="uv run python -m"
SCRIPT="survpfn.scripts.benchmark"

FOLDS=5
TRIALS=20
SEED=42
OUTPUT_DIR="results/benchmark_cr"
LOG_DIR="logs"

EPOCHS=50
LR="1e-4"
DEVICE="cuda:0"
BATCH_SIZE=128
N_ENSEMBLE=1

# ── Dataset groups ────────────────────────────────────────────────────────────
CR_DATASETS="FRAMINGHAM PBC2 SUPPORT_CR SYNTHETIC_CR"
ALL_DATASETS="$CR_DATASETS"

# ── Model groups (must match ALL_MODELS keys in models/__init__.py) ───────────
# ── Model groups (must match analysis.py) ──────────────────────────────────────
CLASSICAL_CR="cox_cr aj_cr fine_gray_cr survival_boost_cr"
DEEP_CR="$CLASSICAL_CR deephit_cr dysurv_cr survtrace_cr"

FM_CR_EMBEDDING="tabpfn_embedding_deephit_cr tabdpt_embedding_deephit_cr tabicl_embedding_deephit_cr tabpfn_embedding_cox_cr tabdpt_embedding_cox_cr tabicl_embedding_cox_cr"
FM_CR_ZEROSHOT="tabpfn_zeroshot_cr_ens tabdpt_zeroshot_cr_ens tabicl_zeroshot_cr_ens"

ALL_CR_MODELS="$DEEP_CR $FM_CR_EMBEDDING" # $FM_CR_ZEROSHOT"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 [group] [dataset | "ds1 ds2 ..."]

Groups (default: all):
  classical_cr              Cox-CR, AJ, Fine-Gray, SurvivalBoost
  deep_cr                   classical_cr + DeepHit-CR
  fm_embedding              FM frozen-embedding DeepHit-CR (all 3 backbones)
  fm_cr_deephit             fm_embedding + fm_joint
  fm_zeroshot               FM zero-shot CR (all 3 backbones)
  tabpfn_cr | tabdpt_cr | tabicl_cr
  all

Datasets (default): $CR_DATASETS
EOF
    exit 0
}

GROUP="${1:-all}"
DATASET_OVERRIDE="${2:-}"

[[ "$GROUP" == "--help" || "$GROUP" == "-h" ]] && usage

case "${DATASET_OVERRIDE:-}" in
    "") DATASETS="$ALL_DATASETS" ;;
    *)  DATASETS="$DATASET_OVERRIDE" ;;
esac

case "$GROUP" in
    all)             MODELS="$ALL_CR_MODELS" ;;
    classical_cr)    MODELS="$CLASSICAL_CR" ;;
    deep_cr)         MODELS="$DEEP_CR" ;;
    fm_embedding)    MODELS="$FM_CR_EMBEDDING" ;;
    fm_joint)        MODELS="$FM_CR_JOINT" ;;
    fm_cr_deephit)   MODELS="$FM_CR_DEEPHIT" ;;
    fm_zeroshot)     MODELS="$FM_CR_ZEROSHOT" ;;
    tabpfn_cr)       MODELS=$(echo "$ALL_CR_MODELS" | xargs -n1 | grep tabpfn | xargs) ;;
    tabdpt_cr)       MODELS=$(echo "$ALL_CR_MODELS" | xargs -n1 | grep tabdpt | xargs) ;;
    tabicl_cr)       MODELS=$(echo "$ALL_CR_MODELS" | xargs -n1 | grep tabicl | xargs) ;;
    *)               MODELS=($GROUP)
esac

read -ra EXTRA <<< "$(_extra_args "$MODELS")"

# ── Run ───────────────────────────────────────────────────────────────────────
printf '\n═══════════════════════════════════════════════════════════════\n'
printf '  run_cr.sh  •  group=%s\n' "$GROUP"
printf '  datasets : %s\n' "$DATASETS"
printf '  folds=%-2s  trials=%-3s  seed=%s\n' "$FOLDS" "$TRIALS" "$SEED"
printf '═══════════════════════════════════════════════════════════════\n'

run_models "$GROUP" "$MODELS" "$DATASETS" "${EXTRA[@]}"

printf '\n═══════════════════════════════════════════════════════════════\n'
printf '  Done. Results: %s   Logs: %s\n' "$OUTPUT_DIR" "$LOG_DIR"
printf '═══════════════════════════════════════════════════════════════\n'
