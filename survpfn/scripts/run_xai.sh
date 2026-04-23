#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_xai.sh — Post-benchmark analysis, tables, and figures for SurvPFN
#
# Usage
# ─────
#   ./run_xai.sh                    # full pipeline (aggregate → all tables/figs)
#   ./run_xai.sh aggregate          # just re-aggregate results
#   ./run_xai.sh tables             # LaTeX tables (public datasets)
#   ./run_xai.sh tables_ehr         # LaTeX tables (EHR datasets)
#   ./run_xai.sh tables_survset     # LaTeX tables (SurvSet)
#   ./run_xai.sh tables_all         # LaTeX tables (all datasets grouped)
#   ./run_xai.sh figures            # comparison bar charts + heatmaps
#   ./run_xai.sh significance       # statistical tests (public)
#   ./run_xai.sh significance_all   # statistical tests (all datasets)
#   ./run_xai.sh analysis           # analysis.py figures (heatmap, rank, etc.)
#   ./run_xai.sh cr                 # competing-risks tables + figures
#   ./run_xai.sh paper              # summary table + significance for paper
#
# Dataset groups
# ──────────────
#   public     → SUPPORT2 METABRIC GBSG WHAS500 VETERANS FLCHAIN SEER  (7)
#   ehr        → EICU_SURV MIMIC_SURV_B  (2)
#   public+ehr → public + ehr  (9)
#   survset    → SS_* datasets  (26)
#   ormoni     → ORMONI_TIRODEI_*  (4)
#   all        → public + ehr + survset  (35)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON="uv run"

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_SR="results/benchmark"
RESULTS_CR="results/benchmark_cr"
FIGURES_SR="results/xai/figures"
FIGURES_CR="results/xai/figures_cr"
TABLES_SR="results/xai/tables"
TABLES_CR="results/xai/tables_cr"
STATS_SR="results/xai/stats"

AGG_SR="$RESULTS_SR/aggregated.csv"
AGG_CR="$RESULTS_CR/aggregated.csv"

CMD="${1:-all}"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: aggregate results into a single CSV
# ─────────────────────────────────────────────────────────────────────────────
do_aggregate() {
    echo "━━━ Aggregating SR results ($RESULTS_SR) …"
    $PYTHON survpfn/xai/aggregate.py \
        --output-dir "$RESULTS_SR" \
        --dest       "$AGG_SR"

    if [[ -d "$RESULTS_CR" ]]; then
        echo "━━━ Aggregating CR results ($RESULTS_CR) …"
        $PYTHON survpfn/xai/aggregate.py \
            --output-dir "$RESULTS_CR" \
            --dest       "$AGG_CR"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX tables
# ─────────────────────────────────────────────────────────────────────────────
do_tables_public() {
    echo "━━━ LaTeX tables — public datasets …"
    $PYTHON survpfn/xai/latex_table.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$TABLES_SR" \
        --dataset-group public \
        --metrics       C_td IBS \
        --group-datasets \
        --summary
}

do_tables_ehr() {
    echo "━━━ LaTeX tables — EHR datasets …"
    $PYTHON survpfn/xai/latex_table.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$TABLES_SR/ehr" \
        --dataset-group ehr \
        --metrics       C_td IBS \
        --summary
}

do_tables_survset() {
    echo "━━━ LaTeX tables — SurvSet datasets …"
    $PYTHON survpfn/xai/latex_table.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$TABLES_SR/survset" \
        --dataset-group survset \
        --metrics       C_td IBS \
        --group-datasets
}

do_tables_ormoni() {
    echo "━━━ LaTeX tables — OrmoniTirodei datasets …"
    $PYTHON survpfn/xai/latex_table.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$TABLES_SR/ormoni" \
        --dataset-group ormoni \
        --metrics       C_td IBS \
        --summary
}

do_tables_all() {
    echo "━━━ LaTeX tables — all datasets grouped …"
    $PYTHON survpfn/xai/latex_table.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$TABLES_SR/all" \
        --dataset-group all \
        --metrics       C_td IBS \
        --group-datasets \
        --summary
}

do_tables_public_ehr() {
    echo "━━━ LaTeX tables — public + EHR grouped …"
    $PYTHON survpfn/xai/latex_table.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$TABLES_SR/public_ehr" \
        --dataset-group public+ehr \
        --metrics       C_td IBS \
        --group-datasets \
        --summary
}


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance
# ─────────────────────────────────────────────────────────────────────────────
do_significance_public() {
    echo "━━━ Significance tests — public datasets …"
    $PYTHON survpfn/scripts/significance.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$STATS_SR/public" \
        --datasets      public \
        --references    cox rsf deephit_single \
        --metric        C_td \
        --alpha         0.05
}

do_significance_ehr() {
    echo "━━━ Significance tests — EHR datasets …"
    $PYTHON survpfn/scripts/significance.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$STATS_SR/ehr" \
        --datasets      ehr \
        --references    cox rsf deephit_single \
        --metric        C_td \
        --alpha         0.05
}

do_significance_all() {
    echo "━━━ Significance tests — all datasets …"
    $PYTHON survpfn/scripts/significance.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$STATS_SR/all" \
        --references    cox rsf deephit_single \
        --metric        C_td \
        --top-n-cd      25 \
        --alpha         0.05
}

do_significance_ibs() {
    echo "━━━ Significance tests — IBS, public datasets …"
    $PYTHON survpfn/scripts/significance.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$STATS_SR/public_ibs" \
        --datasets      public \
        --references    cox rsf deephit_single \
        --metric        IBS \
        --alpha         0.05
}

# ─────────────────────────────────────────────────────────────────────────────
# analysis.py figures (heatmap, rank diagram, etc.)
# ─────────────────────────────────────────────────────────────────────────────
do_analysis() {
    echo "━━━ analysis.py figures …"
    $PYTHON survpfn/xai/analysis.py \
        --results-dir  "$RESULTS_SR" \
        --output-dir   "$FIGURES_SR/analysis" \
		--figures fig02 fig10 fig12
}

do_analysis_mini() {
    echo "━━━ analysis.py figures …"
    $PYTHON survpfn/xai/analysis_mini.py \
        --results-dir  "$RESULTS_SR" \
        --output-dir   "$FIGURES_SR/analysis" \
		--figures fig02 # fig10 fig12
}

# ─────────────────────────────────────────────────────────────────────────────
# Competing-risks tables + figures
# ─────────────────────────────────────────────────────────────────────────────
do_cr() {
    if [[ ! -f "$AGG_CR" ]]; then
        echo "  [WARN] $AGG_CR not found — run aggregate first."
        return
    fi

    echo "━━━ LaTeX tables — competing risks …"
    $PYTHON survpfn/xai/latex_table.py \
        --aggregated   "$AGG_CR" \
        --output-dir   "$TABLES_CR" \
        --datasets      FRAMINGHAM PBC2 SUPPORT_CR SYNTHETIC_CR \
        --metrics       C_td IBS \
        --summary

    echo "━━━ Significance tests — competing risks …"
    $PYTHON survpfn/scripts/significance.py \
        --aggregated   "$AGG_CR" \
        --output-dir   "$STATS_SR/cr" \
        --references    cox_cr aj_cr fine_gray_cr \
        --metric        C_td \
        --no-cd \
        --alpha         0.05
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics table + diversity figures (dataset_stats.py)
# ─────────────────────────────────────────────────────────────────────────────
do_dataset_stats() {
    local group="${1:-all}"
    echo "━━━ Dataset statistics — group: $group …"
    $PYTHON -m survpfn.xai.dataset_stats \
        --group      "$group" \
        --output-dir "results/xai/datasets"
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset EDA figures (analyze_data_comprehensive.py)
# ─────────────────────────────────────────────────────────────────────────────
do_eda() {
    local group="${1:-public}"
    echo "━━━ EDA figures — group: $group …"
    $PYTHON -m survpfn.xai.analyze_data_comprehensive \
        --group      "$group" \
        --output-dir "results/xai/datasets/eda"
}

# ─────────────────────────────────────────────────────────────────────────────
# Correlation / leakage / separation audit (correlation_analysis.py)
# ─────────────────────────────────────────────────────────────────────────────
do_correlation() {
    local group="${1:-public}"
    echo "━━━ Correlation audit — group: $group …"
    $PYTHON -m survpfn.xai.correlation_analysis \
        --group      "$group" \
        --output-dir "results/xai/datasets/correlation/$group"
}

# ─────────────────────────────────────────────────────────────────────────────
# Paper-ready bundle: summary table + public significance (C_td + IBS)
# ─────────────────────────────────────────────────────────────────────────────
do_paper() {
    echo "━━━ Paper bundle …"

    # Compact summary table (public, C_td)
    $PYTHON survpfn/xai/latex_table.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$TABLES_SR/paper" \
        --dataset-group public+ehr \
        --metrics       C_td IBS \
        --group-datasets \
        --summary

    # Significance (C_td, public)
    $PYTHON survpfn/scripts/significance.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$STATS_SR/paper" \
        --datasets      public \
        --references    cox rsf deephit_single \
        --metric        C_td \
        --alpha         0.05

    # Significance (IBS, public)
    $PYTHON survpfn/scripts/significance.py \
        --aggregated   "$AGG_SR" \
        --output-dir   "$STATS_SR/paper_ibs" \
        --datasets      public \
        --references    cox rsf deephit_single \
        --metric        IBS \
        --no-cd \
        --alpha         0.05

    # Rank / heatmap figures
    $PYTHON survpfn/xai/analysis.py \
        --results-dir  "$RESULTS_SR" \
        --output-dir   "$FIGURES_SR/paper" \
		--figures fig02 fig10 fig12

}

# ─────────────────────────────────────────────────────────────────────────────
# Fairness evaluation — model C-index per demographic subgroup
# Requires: benchmark.py --save-predictions has been run first
# ─────────────────────────────────────────────────────────────────────────────
do_fairness() {
    local datasets="${1:-EICU_SURV MIMIC_SURV_B}"
    echo "━━━ Fairness evaluation — $datasets …"
    $PYTHON -m survpfn.xai.fairness_eval \
        --datasets     $datasets \
        --pred-dir     "results/predictions" \
        --output-dir   "results/xai/fairness" \
        --min-n        30
}

do_fairness_eicu() {
    echo "━━━ Fairness evaluation — eICU …"
    $PYTHON -m survpfn.xai.fairness_eval \
        --datasets     EICU_SURV \
        --pred-dir     "results/predictions" \
        --output-dir   "results/xai/fairness"
}

do_fairness_mimic() {
    echo "━━━ Fairness evaluation — MIMIC-IV …"
    $PYTHON -m survpfn.xai.fairness_eval \
        --datasets     MIMIC_SURV_B \
        --pred-dir     "results/predictions" \
        --output-dir   "results/xai/fairness"
}

# ─────────────────────────────────────────────────────────────────────────────
# Demographic subgroup EDA (subgroup_eda.py)
# Targets EHR datasets: age / sex / race / insurance
# ─────────────────────────────────────────────────────────────────────────────
do_subgroup() {
    local datasets="${1:-EICU_SURV MIMIC_SURV_B}"
    echo "━━━ Subgroup EDA — $datasets …"
    $PYTHON -m survpfn.xai.subgroup_eda \
        --datasets     $datasets \
        --output-dir   "results/xai/subgroups"
}

do_subgroup_eicu() {
    echo "━━━ Subgroup EDA — eICU …"
    $PYTHON -m survpfn.xai.subgroup_eda \
        --datasets     EICU_SURV \
        --output-dir   "results/xai/subgroups"
}

do_subgroup_mimic() {
    echo "━━━ Subgroup EDA — MIMIC-IV …"
    $PYTHON -m survpfn.xai.subgroup_eda \
        --datasets     MIMIC_SURV_B \
        --output-dir   "results/xai/subgroups"
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
case "$CMD" in
    aggregate)          do_aggregate ;;
    tables)             do_tables_public ;;
    tables_ehr)         do_tables_ehr ;;
    tables_survset)     do_tables_survset ;;
    tables_ormoni)      do_tables_ormoni ;;
    tables_all)         do_tables_all ;;
    tables_public_ehr)  do_tables_public_ehr ;;
    significance)       do_significance_public ;;
    significance_ehr)   do_significance_ehr ;;
    significance_all)   do_significance_all ;;
    significance_ibs)   do_significance_ibs ;;
    analysis)           do_analysis ;;
    cr)                 do_cr ;;
    paper)              do_paper ;;
    # ── Dataset EDA ──────────────────────────────────────────────────────────
    dataset_stats)          do_dataset_stats "${2:-all}" ;;
    dataset_stats_public)   do_dataset_stats "public" ;;
    dataset_stats_ehr)      do_dataset_stats "ehr" ;;
    dataset_stats_survset)  do_dataset_stats "survset" ;;
    dataset_stats_all)      do_dataset_stats "all" ;;
    eda)                    do_eda "${2:-public}" ;;
    eda_public)             do_eda "public" ;;
    eda_ehr)                do_eda "ehr" ;;
    eda_all)                do_eda "all" ;;
    correlation)            do_correlation "${2:-public}" ;;
    correlation_ehr)        do_correlation "ehr" ;;
    correlation_all)        do_correlation "all" ;;
    # ── Fairness evaluation (model predictions per subgroup) ─────────────────
    fairness)               do_fairness ;;
    fairness_eicu)          do_fairness_eicu ;;
    fairness_mimic)         do_fairness_mimic ;;
    # ── Subgroup EDA (EHR, appendix) ─────────────────────────────────────────
    subgroup)               do_subgroup ;;
    subgroup_eicu)          do_subgroup_eicu ;;
    subgroup_mimic)         do_subgroup_mimic ;;
    # ── Convenience bundles ───────────────────────────────────────────────────
    data)
        do_eda "public"
        do_dataset_stats "all"
        do_correlation "public"
        do_subgroup
        ;;
    all)
        do_aggregate
        do_tables_public
        do_tables_public_ehr
        do_tables_survset
        do_significance_public
        do_significance_all
        do_analysis
        do_dataset_stats "all"
        do_eda "public"
        do_subgroup
        do_fairness
        # do_cr
        ;;
    *)
        echo "Unknown command: $CMD"
        echo ""
        echo "Benchmark results:"
        echo "  aggregate | tables | tables_ehr | tables_survset | tables_ormoni"
        echo "  tables_all | tables_public_ehr"
        echo "  significance | significance_ehr | significance_all | significance_ibs"
        echo "  analysis | cr | paper"
        echo ""
        echo "Dataset EDA:"
        echo "  dataset_stats [group]   — stats table + diversity figures"
        echo "  dataset_stats_public    — public datasets only"
        echo "  dataset_stats_ehr       — EHR datasets only"
        echo "  dataset_stats_survset   — SurvSet only"
        echo "  dataset_stats_all       — all SR datasets"
        echo "  eda [group]             — per-dataset EDA (KM, time dist, features)"
        echo "  eda_public | eda_ehr | eda_all"
        echo "  correlation [group]     — leakage / separation / collinearity audit"
        echo "  correlation_ehr | correlation_all"
        echo ""
        echo "Fairness evaluation (requires --save-predictions benchmark run):"
        echo "  fairness                — eICU + MIMIC-IV (age / sex / race / insurance)"
        echo "  fairness_eicu           — eICU only"
        echo "  fairness_mimic          — MIMIC-IV only"
        echo ""
        echo "Subgroup EDA (EHR, for appendix):"
        echo "  subgroup                — eICU + MIMIC-IV (age / sex / race)"
        echo "  subgroup_eicu           — eICU only"
        echo "  subgroup_mimic          — MIMIC-IV only"
        echo ""
        echo "Bundles:"
        echo "  data | all"
        exit 1
        ;;
esac

echo ""
echo "Done."
