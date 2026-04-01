#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  –  Bulk benchmark runner for survpfn
#
# Usage:
#   ./run.sh                               # run ALL models on ALL datasets
#   ./run.sh classical                     # cox + tree models only
#   ./run.sh deep                          # deep survival baselines only
#   ./run.sh tabpfn                        # TabPFN jointly-trained heads only
#   ./run.sh fm_embedding                  # all FM frozen-embedding × head combos
#   ./run.sh tabpfn_embedding              # TabPFN frozen embedding × 4 heads
#   ./run.sh tabdpt                        # TabDPT frozen embedding × 4 heads
#   ./run.sh tabicl                        # TabICL frozen embedding × 4 heads
#   ./run.sh all GBSG                      # all models, one dataset
#   ./run.sh deep "GBSG METABRIC"          # deep models, two datasets
#   ./run.sh --parallel                    # one background job per dataset
#
# Model groups
# ─────────────────────────────────────────────────────────────────────────────
#   classical      →  cox  km  rsf  gbsa
#   deep           →  deepsurv  mtlr  pchazard  deephit_single
#   tabpfn         →  tabpfn_cox  tabpfn_deephit  tabpfn_pchazard  tabpfn_mtlr
#   tabpfn_embedding → tabpfn_embedding_cox  tabpfn_embedding_deephit
#                      tabpfn_embedding_pchazard  tabpfn_embedding_mtlr
#   tabdpt         →  tabdpt_embedding_{cox,deephit,pchazard,mtlr}
#   tabicl         →  tabicl_embedding_{cox,deephit,pchazard,mtlr}
#   fm_embedding   →  tabpfn_embedding + tabdpt + tabicl  (12 models total)
#   all            →  everything above
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON="uv run"
SCRIPT="survpfn/scripts/benchmark.py"

FOLDS=5
TRIALS=20
SEED=42
OUTPUT_DIR="results/benchmark"
LOG_DIR="logs"

# FM / deep-model GPU args
EPOCHS=100
LR="1e-3"
DEVICE="cuda:0"

# TabDPT checkpoint (override via env or edit here)
TABDPT_CHECKPOINT="${TABDPT_CHECKPOINT:-}"
TABDPT_CONTEXT_SIZE="${TABDPT_CONTEXT_SIZE:-128}"

# ── Dataset groups ────────────────────────────────────────────────────────────
PUBLIC_DATASETS="SUPPORT2 METABRIC GBSG WHAS500 VETERANS FLCHAIN"
ALL_DATASETS="$PUBLIC_DATASETS"

# ── Model groups ──────────────────────────────────────────────────────────────
CLASSICAL_MODELS="cox km rsf gbsa"
DEEP_MODELS="deepsurv mtlr pchazard deephit_single"

TABPFN_JOINT="tabpfn_cox tabpfn_deephit tabpfn_pchazard tabpfn_mtlr"
TABPFN_EMBEDDING="tabpfn_embedding_cox tabpfn_embedding_deephit tabpfn_embedding_pchazard tabpfn_embedding_mtlr"
TABDPT_MODELS="tabdpt_embedding_cox tabdpt_embedding_deephit tabdpt_embedding_pchazard tabdpt_embedding_mtlr"
TABICL_MODELS="tabicl_embedding_cox tabicl_embedding_deephit tabicl_embedding_pchazard tabicl_embedding_mtlr"
FM_EMBEDDING="$TABPFN_EMBEDDING $TABDPT_MODELS $TABICL_MODELS"

ALL_MODELS="$CLASSICAL_MODELS $DEEP_MODELS $TABPFN_JOINT $FM_EMBEDDING"

# ── Helpers ───────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [group] [\"dataset1 dataset2 ...\"] [--parallel]"
    echo ""
    echo "Groups:    all (default) | classical | deep | tabpfn | tabpfn_embedding"
    echo "           tabdpt | tabicl | fm_embedding"
    echo "Datasets:  $PUBLIC_DATASETS"
    echo "Flags:     --parallel   run each dataset as a background job"
    exit 0
}

timestamp() { date +"%Y%m%d_%H%M%S"; }

# _extra_args: emit GPU + TabDPT args when the model set includes FM models
_extra_args() {
    local models="$1"
    local args=()
    if echo "$models" | grep -qE "tabpfn_|tabdpt_|tabicl_"; then
        args+=("--epochs" "$EPOCHS" "--lr" "$LR" "--device" "$DEVICE")
    fi
    if echo "$models" | grep -q "tabdpt_" && [[ -n "$TABDPT_CHECKPOINT" ]]; then
        args+=("--tabdpt-checkpoint" "$TABDPT_CHECKPOINT"
               "--tabdpt-context-size" "$TABDPT_CONTEXT_SIZE")
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

for arg in "$@"; do
    [[ "$arg" == "--parallel" ]] && PARALLEL=true
done

[[ "$GROUP" == "--help" || "$GROUP" == "-h" ]] && usage
[[ "$GROUP" == "--parallel" ]] && GROUP="all"

DATASETS="${DATASET_OVERRIDE:-$ALL_DATASETS}"

case "$GROUP" in
    all)              MODELS="$ALL_MODELS" ;;
    classical)        MODELS="$CLASSICAL_MODELS" ;;
    deep)             MODELS="$DEEP_MODELS" ;;
    tabpfn)           MODELS="$TABPFN_JOINT" ;;
    tabpfn_embedding) MODELS="$TABPFN_EMBEDDING" ;;
    tabdpt)           MODELS="$TABDPT_MODELS" ;;
    tabicl)           MODELS="$TABICL_MODELS" ;;
    fm_embedding)     MODELS="$FM_EMBEDDING" ;;
    *) echo "Unknown group: $GROUP"; usage ;;
esac

# Build extra args for this model set
read -ra EXTRA <<< "$(_extra_args "$MODELS")"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  survpfn benchmark  •  group=$GROUP  •  parallel=$PARALLEL"
echo "  datasets : $DATASETS"
echo "  folds=$FOLDS  trials=$TRIALS  seed=$SEED"
echo "═══════════════════════════════════════════════════════════════"

if $PARALLEL; then
    echo ""
    echo "Parallel mode — spawning one job per dataset …"
    echo ""
    for ds in $DATASETS; do
        run_dataset_parallel "$GROUP" "$MODELS" "$ds" "${EXTRA[@]}"
    done
    echo ""
    echo "All jobs spawned. Waiting for completion …"
    wait
    echo "All parallel jobs done."
else
    run_models "$GROUP" "$MODELS" "$DATASETS" "${EXTRA[@]}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done. Results in: $OUTPUT_DIR"
echo "  Logs  in: $LOG_DIR"
echo "═══════════════════════════════════════════════════════════════"
