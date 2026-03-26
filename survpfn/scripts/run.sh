#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  –  Bulk benchmark runner for survpfn
#
# Usage:
#   ./run.sh                          # run ALL models on ALL datasets
#   ./run.sh classical                # cox + tree models only
#   ./run.sh deep                     # deep survival models only
#   ./run.sh tabpfn                   # TabPFN-Surv models only
#   ./run.sh all GBSG                 # all models, one dataset
#   ./run.sh deep "GBSG METABRIC"     # deep models, two datasets
#   ./run.sh --parallel               # run each dataset in background in parallel
#
# Model groups:
#   classical  →  cox  km  rsf  gbsa
#   deep       →  deepsurv  mtlr  pchazard  deephit_single  embedding_cox
#   tabpfn     →  surv_cox  surv_deephit  surv_pchazard  surv_mtlr
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON="uv run"
SCRIPT="survpfn/scripts/benchmark.py"

FOLDS=5
TRIALS=20
SEED=42
OUTPUT_DIR="results"
LOG_DIR="logs"

# TabPFN-specific
EPOCHS=50
LR="1e-3"
DEVICE="cuda:0"

# ── Dataset groups ────────────────────────────────────────────────────────────
STANDARD_DATASETS="SUPPORT2 METABRIC GBSG SIRBU"
SIRBU_TASKS="SIRBU_mortality SIRBU_cv SIRBU_mi SIRBU_stroke"
ALL_DATASETS="$STANDARD_DATASETS $SIRBU_TASKS"

# ── Model groups ──────────────────────────────────────────────────────────────
CLASSICAL_MODELS="cox km rsf gbsa"
DEEP_MODELS="deepsurv mtlr pchazard deephit_single embedding_cox"
TABPFN_MODELS="surv_cox surv_deephit surv_pchazard surv_mtlr"
ALL_MODELS="$CLASSICAL_MODELS $DEEP_MODELS $TABPFN_MODELS"

# ── Helpers ───────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [group] [\"dataset1 dataset2 ...\"] [--parallel]"
    echo ""
    echo "Groups:    all (default) | classical | deep | tabpfn"
    echo "Datasets:  SUPPORT2 METABRIC GBSG SIRBU"
    echo "           SIRBU_mortality SIRBU_cv SIRBU_mi SIRBU_stroke"
    echo "Flags:     --parallel  run each dataset as a background job"
    exit 0
}

timestamp() { date +"%Y%m%d_%H%M%S"; }

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

# run_dataset_parallel <label> <models> <dataset> [extra_args...]
run_dataset_parallel() {
    local label="$1"
    local models="$2"
    local dataset="$3"
    shift 3
    local extra=("$@")

    mkdir -p "$LOG_DIR"
    local logfile="$LOG_DIR/${label}_${dataset}_$(timestamp).log"

    echo "  → Spawning: $dataset ($label) — log: $logfile"

    # shellcheck disable=SC2086
    $PYTHON $SCRIPT \
        --datasets "$dataset" \
        --models $models \
        --folds "$FOLDS" \
        --tune --trials "$TRIALS" \
        --seed "$SEED" \
        --output-dir "$OUTPUT_DIR" \
        "${extra[@]}" \
        > "$logfile" 2>&1 &
}

# ── Parse args ────────────────────────────────────────────────────────────────
GROUP="${1:-all}"
DATASET_OVERRIDE="${2:-}"
PARALLEL=false

# Detect --parallel anywhere in args
for arg in "$@"; do
    [[ "$arg" == "--parallel" ]] && PARALLEL=true
done

[[ "$GROUP" == "--help" || "$GROUP" == "-h" ]] && usage
[[ "$GROUP" == "--parallel" ]] && GROUP="all"

DATASETS="${DATASET_OVERRIDE:-$ALL_DATASETS}"

# Pick model list for selected group
case "$GROUP" in
    all)       MODELS="$ALL_MODELS" ;;
    classical) MODELS="$CLASSICAL_MODELS" ;;
    deep)      MODELS="$DEEP_MODELS" ;;
    tabpfn)    MODELS="$TABPFN_MODELS" ;;
    *) echo "Unknown group: $GROUP"; usage ;;
esac

# ── Run ───────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  survpfn benchmark  •  group=$GROUP  •  parallel=$PARALLEL"
echo "  datasets : $DATASETS"
echo "  folds=$FOLDS  trials=$TRIALS  seed=$SEED"
echo "═══════════════════════════════════════════════════════════════"

# TabPFN models need extra GPU args
tabpfn_extra=("--epochs" "$EPOCHS" "--lr" "$LR" "--device" "$DEVICE")

# Decide whether this group needs tabpfn_extra
needs_tabpfn_extra() {
    # returns 0 (true) if $1 contains any surv_* model
    echo "$1" | grep -q "surv_"
}

if $PARALLEL; then
    # ── Parallel mode: one background job per dataset ──────────────────────
    echo ""
    echo "Parallel mode — spawning one job per dataset …"
    echo ""

    for ds in $DATASETS; do
        if needs_tabpfn_extra "$MODELS" 2>/dev/null; then
            run_dataset_parallel "$GROUP" "$MODELS" "$ds" "${tabpfn_extra[@]}"
        else
            run_dataset_parallel "$GROUP" "$MODELS" "$ds"
        fi
    done

    echo ""
    echo "All jobs spawned. Waiting for completion …"
    wait
    echo "All parallel jobs done."

else
    # ── Sequential mode ────────────────────────────────────────────────────
    if needs_tabpfn_extra "$MODELS" 2>/dev/null; then
        run_models "$GROUP" "$MODELS" "$DATASETS" "${tabpfn_extra[@]}"
    else
        run_models "$GROUP" "$MODELS" "$DATASETS"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done. Results in: $OUTPUT_DIR"
echo "  Logs  in: $LOG_DIR"
echo "═══════════════════════════════════════════════════════════════"
