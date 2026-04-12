#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sr.sh — Single-risk survival benchmark runner
#
# Usage
# ─────
#   ./run_sr.sh                            # all models, all datasets
#   ./run_sr.sh classical                  # cox + tree baselines only
#   ./run_sr.sh deep                       # deep survival baselines
#   ./run_sr.sh fm_embedding               # all FM frozen-embedding models
#   ./run_sr.sh fm_joint                   # all FM jointly-trained models
#   ./run_sr.sh fm                         # all FM models (embedding + joint + zeroshot)
#   ./run_sr.sh tabpfn                     # all TabPFN variants
#   ./run_sr.sh tabdpt                     # all TabDPT variants
#   ./run_sr.sh tabicl                     # all TabICL variants
#   ./run_sr.sh zeroshot                   # zero-shot ICL (single_context mode)
#   ./run_sr.sh zeroshot_perbin            # zero-shot ICL (per_bin mode)
#   ./run_sr.sh surv_adapter               # KM-adapter models
#   ./run_sr.sh all GBSG                   # all SR models, one dataset
#   ./run_sr.sh deep "GBSG METABRIC"       # deep models, two datasets
#   ./run_sr.sh classical public           # classical models, public datasets
#   ./run_sr.sh all --parallel             # one background job per dataset
#
# Dataset keywords (2nd positional arg)
# ──────────────────────────────────────
#   public    → SUPPORT2 METABRIC GBSG WHAS500 VETERANS FLCHAIN SEER
#   survset   → 26 curated SS_* datasets
#   ormoni_tirodei → ORMONI_TIRODEI_CV ORMONI_TIRODEI_MI ORMONI_TIRODEI_STROKE ORMONI_TIRODEI_MORTALITY
#   ehr       → EICU_SURV MIMIC_SURV_B  (large-scale ICU survival)
#   (default) → public + survset
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
OUTPUT_DIR="results/benchmark"
LOG_DIR="logs"

EPOCHS=50
LR="1e-4"
DEVICE="cuda:0"
BATCH_SIZE=64
N_ENSEMBLE=5

# ── Dataset groups ────────────────────────────────────────────────────────────
PUBLIC_DATASETS="SUPPORT2 METABRIC GBSG WHAS500 VETERANS FLCHAIN SEER"
ORMONI_TIRODEI_DATASETS="ORMONI_TIRODEI_CV ORMONI_TIRODEI_MI ORMONI_TIRODEI_STROKE ORMONI_TIRODEI_MORTALITY"
URRAH_DATASETS="URRAH"
SIRBU_DATASETS="ORMONI_TIRODEI_MORTALITY" #URRAH
SURVSET_DATASETS="SS_CANCER SS_BREAST SS_GBSG2 SS_ROTT2 SS_COLON SS_PROSTATE \
SS_OVARIAN SS_MELANOMA SS_E1684 SS_PBC SS_HEPATOCELLULAR SS_NWTCO \
SS_RETINOPATHY SS_HEART SS_CGD SS_COST SS_LEUKSURV SS_DIALYSIS \
SS_ACTG SS_RHC SS_VLBW SS_GRACE SS_TRACE SS_DLBCL SS_DIABETES SS_FRAMINGHAM"
EHR_DATASETS="EICU_SURV MIMIC_SURV_B"
ALL_DATASETS="$PUBLIC_DATASETS $SURVSET_DATASETS"

# ── Model groups (must match analysis.py groupings) ───────────────────────────
CLASSICAL_MODELS="km cox"
TREE_MODELS="rsf gbsa"
DEEP_MODELS="deepsurv mtlr pchazard deephit_single survtrace"

EMBEDDING_COX="tabpfn_embedding_cox tabdpt_embedding_cox tabicl_embedding_cox"
EMBEDDING_DEEPHIT="tabpfn_embedding_deephit tabdpt_embedding_deephit tabicl_embedding_deephit"
EMBEDDING_MTLR="tabpfn_embedding_mtlr tabdpt_embedding_mtlr tabicl_embedding_mtlr"
EMBEDDING_PCHAZARD="tabpfn_embedding_pchazard tabdpt_embedding_pchazard tabicl_embedding_pchazard"
FM_EMBEDDING="$EMBEDDING_COX $EMBEDDING_DEEPHIT $EMBEDDING_MTLR $EMBEDDING_PCHAZARD"

JOINT_COX="tabpfn_joint_cox tabdpt_joint_cox tabicl_joint_cox"
JOINT_DEEPHIT="tabpfn_joint_deephit tabdpt_joint_deephit tabicl_joint_deephit"
JOINT_MTLR="tabpfn_joint_mtlr tabdpt_joint_mtlr tabicl_joint_mtlr"
JOINT_PCHAZARD="tabpfn_joint_pchazard tabdpt_joint_pchazard tabicl_joint_pchazard"
FM_JOINT="$JOINT_COX $JOINT_DEEPHIT $JOINT_MTLR $JOINT_PCHAZARD"

ZEROSHOT_MODELS="tabpfn_zeroshot tabdpt_zeroshot tabicl_zeroshot tabpfn_zeroshot_perbin tabdpt_zeroshot_perbin tabicl_zeroshot_perbin"
ZEROSHOT_TEMPORAL="tabpfn_zeroshot_perbin_time tabdpt_zeroshot_perbin_time tabicl_zeroshot_perbin_time tabpfn_zeroshot_perbin_time_ens tabdpt_zeroshot_perbin_time_ens tabicl_zeroshot_perbin_time_ens"
FINETUNE="tabpfn_finetune tabdpt_finetune tabicl_finetune"

ALL_SR_MODELS="$CLASSICAL_MODELS $TREE_MODELS $DEEP_MODELS $FM_EMBEDDING $FM_JOINT $ZEROSHOT_MODELS $ZEROSHOT_TEMPORAL $FINETUNE"

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 [group] [dataset-keyword | "ds1 ds2 ..."] [--parallel]

Groups (default: all):
  classical | deep | fm | fm_embedding | fm_joint
  tabpfn | tabdpt | tabicl
  tabpfn_embedding | tabdpt_embedding | tabicl_embedding
  tabpfn_joint    | tabdpt_joint    | tabicl_joint
  zeroshot | zeroshot_temporal | finetune | all

Dataset keywords:  public  survset  ormoni_tirodei  ehr  (default: public+survset)
Flags:  --parallel   run one background job per dataset
EOF
    exit 0
}

GROUP="${1:-all}"
DATASET_OVERRIDE="${2:-}"
PARALLEL=false

for arg in "$@"; do [[ "$arg" == "--parallel" ]] && PARALLEL=true; done
[[ "$GROUP" == "--help" || "$GROUP" == "-h" ]] && usage
[[ "$GROUP" == "--parallel" ]] && GROUP="all"

case "${DATASET_OVERRIDE:-}" in
    public)        DATASETS="$PUBLIC_DATASETS" ;;
    survset)       DATASETS="$SURVSET_DATASETS" ;;
    ormoni_tirodei) DATASETS="$ORMONI_TIRODEI_DATASETS" ;;
    sirbu)         DATASETS="$SIRBU_DATASETS" ;;
    ehr)           DATASETS="$EHR_DATASETS" ;;
    "")            DATASETS="$ALL_DATASETS" ;;
    *)             DATASETS="$DATASET_OVERRIDE" ;;
esac

case "$GROUP" in
    all)               MODELS="$ALL_SR_MODELS" ;;
    classical)         MODELS="$CLASSICAL_MODELS" ;;
    tree)              MODELS="$TREE_MODELS" ;;
    deep)              MODELS="$DEEP_MODELS" ;;
    fm)                MODELS="$FM_EMBEDDING $FM_JOINT $ZEROSHOT_MODELS $ZEROSHOT_TEMPORAL $FINETUNE" ;;
    fm_embedding)      MODELS="$FM_EMBEDDING" ;;
    fm_joint)          MODELS="$FM_JOINT" ;;
    embedding_cox)     MODELS="$EMBEDDING_COX" ;;
    embedding_deephit) MODELS="$EMBEDDING_DEEPHIT" ;;
    embedding_pch)     MODELS="$EMBEDDING_PCHAZARD" ;;
    embedding_mtlr)     MODELS="$EMBEDDING_MTLR" ;;
    joint_cox)         MODELS="$JOINT_COX" ;;
    joint_deephit)     MODELS="$JOINT_DEEPHIT" ;;
    zeroshot)          MODELS="$ZEROSHOT_MODELS" ;;
    zeroshot_temporal) MODELS="$ZEROSHOT_TEMPORAL" ;;
    finetune)          MODELS="$FINETUNE" ;;
    tabpfn)            MODELS=$(echo "$ALL_SR_MODELS" | xargs -n1 | grep tabpfn | xargs) ;;
    tabdpt)            MODELS=$(echo "$ALL_SR_MODELS" | xargs -n1 | grep tabdpt | xargs) ;;
    tabicl)            MODELS=$(echo "$ALL_SR_MODELS" | xargs -n1 | grep tabicl | xargs) ;;
    *)                 MODELS="$GROUP" ;;
esac

read -ra EXTRA <<< "$(_extra_args "$MODELS")"

# ── Run ───────────────────────────────────────────────────────────────────────
printf '\n═══════════════════════════════════════════════════════════════\n'
printf '  run_sr.sh  •  group=%-12s  parallel=%s\n' "$GROUP" "$PARALLEL"
printf '  datasets : %s\n' "$DATASETS"
printf '  folds=%-2s  trials=%-3s  seed=%s\n' "$FOLDS" "$TRIALS" "$SEED"
printf '═══════════════════════════════════════════════════════════════\n'

if $PARALLEL; then
    for ds in $DATASETS; do
        run_dataset_parallel "$GROUP" "$MODELS" "$ds" "${EXTRA[@]}"
    done
    echo "All jobs spawned. Waiting …"
    wait
    echo "Done."
else
    run_models "$GROUP" "$MODELS" "$DATASETS" "${EXTRA[@]}"
fi

printf '\n═══════════════════════════════════════════════════════════════\n'
printf '  Done. Results: %s   Logs: %s\n' "$OUTPUT_DIR" "$LOG_DIR"
printf '═══════════════════════════════════════════════════════════════\n'
