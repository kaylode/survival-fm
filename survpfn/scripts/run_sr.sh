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
#   deep           →  deepsurv  mtlr  pchazard  deephit_single  soden
#   tabpfn         →  tabpfn_cox  tabpfn_deephit  tabpfn_pchazard  tabpfn_mtlr
#   tabpfn_embedding → tabpfn_embedding_cox  tabpfn_embedding_deephit
#                      tabpfn_embedding_pchazard  tabpfn_embedding_mtlr
#   tabdpt         →  tabdpt_embedding_{cox,deephit,pchazard,mtlr}  (frozen)
#   tabdpt_joint   →  tabdpt_{cox,deephit,pchazard,mtlr}            (jointly-trained)
#   tabicl         →  tabicl_embedding_{cox,deephit,pchazard,mtlr}  (frozen)
#   tabicl_joint   →  tabicl_{cox,deephit,pchazard,mtlr}            (jointly-trained)
#   fm_embedding   →  tabpfn_embedding + tabdpt + tabicl  (12 models total)
#   beta           →  beta_surv  (trainable backbone + frozen TabPFN)
#   all            →  everything above
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON="uv run python -m"
SCRIPT="survpfn.scripts.benchmark"

FOLDS=5
TRIALS=20
SEED=42
OUTPUT_DIR="results/benchmark"
LOG_DIR="logs"

# FM / deep-model GPU args
EPOCHS=500
LR="1e-3"
DEVICE="cuda:0"

# ── Dataset groups ────────────────────────────────────────────────────────────
PUBLIC_DATASETS="SUPPORT2 METABRIC GBSG WHAS500 VETERANS FLCHAIN SEER"
SIRBU_DATASETS="SIRBU_CV SIRBU_MI SIRBU_STROKE SIRBU_MORTALITY"
# SurvSet healthcare-only benchmark (SS_ prefix) SS_CANCER SS_BREAST SS_GBSG2 SS_ROTT2 SS_COLON SS_PROSTATE SS_OVARIAN SS_MELANOMA SS_E1684 SS_PBC SS_HEPATOCELLULAR SS_NWTCO SS_RETINOPATHY SS_HEART SS_MGUS 
SURVSET_DATASETS="SS_CGD SS_COST SS_LEUKSURV SS_DIALYSIS SS_ACTG SS_RHC SS_VLBW SS_GRACE SS_TRACE  SS_DLBCL SS_DIABETES  SS_FRAMINGHAM"
# Small SurvSet subset (6 datasets for fast sanity checks)
SURVSET_SMALL="SS_VETERAN SS_RETINOPATHY SS_CANCER SS_PBC SS_CGD SS_HEPATOCELLULAR"
ALL_DATASETS="$PUBLIC_DATASETS $SURVSET_DATASETS" #$SIRBU_DATASETS

# ── Model groups ──────────────────────────────────────────────────────────────
CLASSICAL_MODELS="cox km rsf gbsa"
DEEP_MODELS="deepsurv mtlr pchazard deephit_single survtrace" #soden"
# DEEP_MODELS="sur/vtrace" # survtrace soden"

TABPFN_EMBEDDING="tabpfn_embedding_cox  tabpfn_embedding_deephit  tabpfn_embedding_pchazard  tabpfn_embedding_mtlr"
TABPFN_JOINT="tabpfn_joint_cox tabpfn_joint_deephit tabpfn_joint_pchazard tabpfn_joint_mtlr"
TABPFN_MODELS="$TABPFN_EMBEDDING $TABPFN_JOINT tabpfn_zeroshot"

TABDPT_EMBEDDING="tabdpt_embedding_cox tabdpt_embedding_deephit tabdpt_embedding_pchazard tabdpt_embedding_mtlr"
TABDPT_JOINT="tabdpt_joint_cox tabdpt_joint_deephit tabdpt_joint_pchazard tabdpt_joint_mtlr"
TABDPT_MODELS="$TABDPT_EMBEDDING $TABDPT_JOINT tabdpt_zeroshot"
# TABDPT_MODELS="tabdpt_joint_cox_adapter"

TABICL_EMBEDDING="tabicl_embedding_cox tabicl_embedding_deephit tabicl_embedding_pchazard tabicl_embedding_mtlr"
TABICL_JOINT="tabicl_joint_cox tabicl_joint_deephit tabicl_joint_pchazard tabicl_joint_mtlr"
TABICL_MODELS="$TABICL_EMBEDDING $TABICL_JOINT tabicl_zeroshot"

FM_EMBEDDING="$TABPFN_EMBEDDING $TABDPT_EMBEDDING $TABICL_EMBEDDING"
FM_JOINT="$TABPFN_JOINT $TABDPT_JOINT $TABICL_JOINT"
ZEROSHOT_MODELS="tabpfn_zeroshot tabdpt_zeroshot tabicl_zeroshot"
ZEROSHOT_PERBIN="tabpfn_zeroshot_perbin tabdpt_zeroshot_perbin tabicl_zeroshot_perbin"
BETA_MODELS="beta_surv"

FM_MODELS="$FM_EMBEDDING $FM_JOINT $ZEROSHOT_MODELS"
ALL_MODELS="$CLASSICAL_MODELS $DEEP_MODELS $FM_MODELS $BETA_MODELS"

# ── Helpers ───────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [group] [\"dataset1 dataset2 ...\"] [--parallel]"
    echo ""
    echo "Groups:    all (default) | classical | deep | tabpfn | tabdpt | tabicl | fm"
    echo "Datasets:  $PUBLIC_DATASETS"
    echo "Flags:     --parallel   run each dataset as a background job"
    exit 0
}

timestamp() { date +"%Y%m%d_%H%M%S"; }

# _extra_args: emit GPU + TabDPT args when the model set includes FM models
_extra_args() {
    local models="$1"
    local args=()
    if echo "$models" | grep -qE "tabpfn|tabdpt|tabicl|deepsurv|mtlr|pchazard|deephit"; then
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

# Expand dataset group keywords in DATASET_OVERRIDE
case "${DATASET_OVERRIDE:-}" in
    survset)          DATASETS="$SURVSET_DATASETS" ;;
    survset_small)    DATASETS="$SURVSET_SMALL" ;;
    public)           DATASETS="$PUBLIC_DATASETS" ;;
    sirbu)            DATASETS="$SIRBU_DATASETS" ;;
    "")               DATASETS="$ALL_DATASETS" ;;
    *)                DATASETS="$DATASET_OVERRIDE" ;;
esac

case "$GROUP" in
    all)              MODELS="$ALL_MODELS" ;;
    classical)        MODELS="$CLASSICAL_MODELS" ;;
    deep)             MODELS="$DEEP_MODELS" ;;
    tabpfn)           MODELS="$TABPFN_MODELS" ;;
    tabdpt)           MODELS="$TABDPT_MODELS" ;;
    tabicl)           MODELS="$TABICL_MODELS" ;;
    fm_embedding)     MODELS="$FM_EMBEDDING" ;;
    fm)               MODELS="$FM_MODELS" ;;
    fm_joint)         MODELS="$FM_JOINT" ;;
    zeroshot)         MODELS="$ZEROSHOT_MODELS" ;;
    zeroshot_perbin)  MODELS="$ZEROSHOT_PERBIN" ;;
    beta)             MODELS="$BETA_MODELS" ;;
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
