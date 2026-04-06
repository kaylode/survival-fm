#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_competing.sh  –  Bulk benchmark runner for survpfn competing risks
#
# Usage:
#   ./run_competing.sh                               # run ALL models on ALL datasets
#   ./run_competing.sh all FRAMINGHAM                # all models, one dataset
#
# Model groups
# ─────────────────────────────────────────────────────────────────────────────
#   tabpfn_cr      →  tabpfn_{embedding,joint}_{deephit,deephit_v2,cox}_cr
#   tabdpt_cr      →  tabdpt_{embedding,joint}_{deephit,deephit_v2,cox}_cr
#   tabicl_cr      →  tabicl_{embedding,joint}_{deephit,deephit_v2,cox}_cr
#   deep_cr        →  deephit_cr
#   classical_cr   →  cox_cr
#   all            →  everything above
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON="uv run python -m"
SCRIPT="survpfn.scripts.benchmark"

FOLDS=5
TRIALS=20
SEED=42
OUTPUT_DIR="results/benchmark_cr"
LOG_DIR="logs"

# FM / deep-model GPU args
EPOCHS=500
LR="1e-3"
DEVICE="cuda:0"

# ── Dataset groups ────────────────────────────────────────────────────────────
# Matching CR_DATASETS in survpfn/dataloaders/__init__.py
CR_DATASETS="FRAMINGHAM PBC2 SUPPORT_CR SYNTHETIC_CR"
ALL_DATASETS="$CR_DATASETS"

# ── Model groups ──────────────────────────────────────────────────────────────
CLASSICAL_CR_MODELS="cox_cr"
DEEP_CR_MODELS="deephit_cr"

TABPFN_CR_EMBEDDING="tabpfn_embedding_deephit_v2_cr tabpfn_embedding_deephit_v2_cr_adapter tabpfn_embedding_cox_cr tabpfn_embedding_cox_cr_adapter"
TABPFN_CR_JOINT="tabpfn_joint_deephit_v2_cr tabpfn_joint_deephit_v2_cr_adapter tabpfn_joint_cox_cr tabpfn_joint_cox_cr_adapter"
TABPFN_CR_ZEROSHOT="tabpfn_zeroshot_cr_multinomial tabpfn_zeroshot_cr_perevent"
TABPFN_CR="$TABPFN_CR_EMBEDDING $TABPFN_CR_JOINT $TABPFN_CR_ZEROSHOT"

TABDPT_CR_EMBEDDING="tabdpt_embedding_deephit_v2_cr tabdpt_embedding_deephit_v2_cr_adapter tabdpt_embedding_cox_cr tabdpt_embedding_cox_cr_adapter"
TABDPT_CR_JOINT="tabdpt_joint_deephit_v2_cr tabdpt_joint_deephit_v2_cr_adapter tabdpt_joint_cox_cr tabdpt_joint_cox_cr_adapter"
TABDPT_CR="$TABDPT_CR_EMBEDDING $TABDPT_CR_JOINT"

TABICL_CR_EMBEDDING="tabicl_embedding_deephit_v2_cr tabicl_embedding_deephit_v2_cr_adapter tabicl_embedding_cox_cr tabicl_embedding_cox_cr_adapter"
TABICL_CR_JOINT="tabicl_joint_deephit_v2_cr tabicl_joint_deephit_v2_cr_adapter tabicl_joint_cox_cr tabicl_joint_cox_cr_adapter"
TABICL_CR="$TABICL_CR_EMBEDDING $TABICL_CR_JOINT"

ALL_MODELS="$CLASSICAL_CR_MODELS $DEEP_CR_MODELS $TABPFN_CR $TABDPT_CR $TABICL_CR"

# ── Helpers ───────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [group] [\"dataset1 dataset2 ...\"]"
    echo ""
    echo "Groups:    all (default) | classical_cr | deep_cr | tabpfn_cr | tabdpt_cr | tabicl_cr"
    echo "Datasets:  $ALL_DATASETS"
    exit 0
}

timestamp() { date +"%Y%m%d_%H%M%S"; }

# _extra_args: emit GPU
_extra_args() {
    local models="$1"
    local args=()
    if echo "$models" | grep -qE "tabpfn|tabdpt|tabicl|deephit"; then
        args+=("--epochs" "$EPOCHS" "--lr" "$LR" "--device" "$DEVICE")
    fi
    echo "${args[@]}"
}

# run_models <label> <models> <datasets> [extra_args...]
run_models() {
    local label="$1"
    local models="$2"
    local datasets="$3"
    shift 3
    local extra=("$@")

    mkdir -p "$LOG_DIR"
    local logfile="$LOG_DIR/${label}_$(timestamp).log"

    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    printf "│  %-61s│\n" "Group   : $label"
    printf "│  %-61s│\n" "Models  : $models"
    printf "│  %-61s│\n" "Data    : $datasets"
    printf "│  %-61s│\n" "Log     : $logfile"
    echo "└─────────────────────────────────────────────────────────────┘"

    # shellcheck disable=SC2086
    $PYTHON $SCRIPT \
        --datasets $datasets \
        --models $models \
        --folds "$FOLDS" \
        --tune --trials "$TRIALS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        "${extra[@]}" \
        2>&1 | tee "$logfile"
}

# ── Parse args ────────────────────────────────────────────────────────────────
GROUP="${1:-all}"
DATASET_OVERRIDE="${2:-}"

[[ "$GROUP" == "--help" || "$GROUP" == "-h" ]] && usage

case "${DATASET_OVERRIDE:-}" in
    cr|competing)     DATASETS="$CR_DATASETS" ;;
    "")               DATASETS="$ALL_DATASETS" ;;
    *)                DATASETS="$DATASET_OVERRIDE" ;;
esac

case "$GROUP" in
    all)              MODELS="$ALL_MODELS" ;;
    classical_cr)     MODELS="$CLASSICAL_CR_MODELS" ;;
    deep_cr)          MODELS="$DEEP_CR_MODELS" ;;
    tabpfn_cr)        MODELS="$TABPFN_CR" ;;
    tabdpt_cr)        MODELS="$TABDPT_CR" ;;
    tabicl_cr)        MODELS="$TABICL_CR" ;;
    *) echo "Unknown group: $GROUP"; usage ;;
esac

# Build extra args for this model set
read -ra EXTRA <<< "$(_extra_args "$MODELS")"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  survpfn benchmark CR  •  group=$GROUP"
echo "  datasets : $DATASETS"
echo "  folds=$FOLDS  trials=$TRIALS  seed=$SEED"
echo "═══════════════════════════════════════════════════════════════"

run_models "$GROUP" "$MODELS" "$DATASETS" "${EXTRA[@]}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done. Results in: $OUTPUT_DIR"
echo "  Logs  in: $LOG_DIR"
echo "═══════════════════════════════════════════════════════════════"
