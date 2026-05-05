#!/usr/bin/env bash
# =============================================================================
# run_statistical.sh  —  Statistical significance tests for SurvPFN benchmark
#
# Usage:
#   ./run_statistical.sh                         # defaults: all public datasets
#   ./run_statistical.sh --metric IBS            # use IBS instead of C-index
#   ./run_statistical.sh --alpha 0.10            # relaxed threshold
#   ./run_statistical.sh --top-n 15 --no-cd      # no CD diagram, top-15 models
#   ./run_statistical.sh --aggregated path/to/aggregated.csv
#   ./run_statistical.sh --references rsf cox deephit_single
#   ./run_statistical.sh --help
#
# Outputs (written to OUTPUT_DIR):
#   significance_pairwise.csv    — all pairwise Wilcoxon results
#   significance_vs_baselines.csv — each model vs each reference baseline
#   significance_friedman.csv    — Friedman test + average ranks
#   significance_nemenyi.csv     — Nemenyi / Bonferroni-Wilcoxon p-matrix
#   significance_ranks.csv       — per-dataset per-model average ranks
#   significance_summary.txt     — human-readable summary
#   cd_diagram.pdf               — Demšar critical-difference diagram
# =============================================================================
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
PYTHON="uv run"
SCRIPT="survpfn/scripts/significance.py"

AGGREGATED="results/benchmark/aggregated.csv"
OUTPUT_DIR="results/benchmark"
METRIC="C_td"
ALPHA="0.0625"
REFERENCES="rsf cox deephit_single"
TOP_N_CD=20
NO_CD=false

# ── Help ──────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run statistical significance tests on SurvPFN benchmark aggregated results.

Options:
  --aggregated FILE   Path to aggregated.csv  [default: $AGGREGATED]
  --output-dir DIR    Where to write output files  [default: $OUTPUT_DIR]
  --metric METRIC     Metric column to test on  [default: $METRIC]
                      Options: C-index | IBS | AUC | D-cal
  --alpha FLOAT       Significance threshold  [default: $ALPHA]
  --references LIST   Space-separated reference model names
                      [default: $REFERENCES]
  --top-n INT         Max models shown in CD diagram  [default: $TOP_N_CD]
  --no-cd             Skip critical-difference diagram
  --all-metrics       Run once per metric (C-index, IBS, AUC)
  -h, --help          Show this message

Examples:
  # Default run (C-index, α=0.05)
  ./run_statistical.sh

  # Test with IBS calibration metric
  ./run_statistical.sh --metric IBS

  # Compare vs RSF only, relaxed α
  ./run_statistical.sh --references rsf --alpha 0.10

  # Run for all three metrics sequentially
  ./run_statistical.sh --all-metrics
EOF
    exit 0
}

# ── Parse args ────────────────────────────────────────────────────────────────
ALL_METRICS=false
REFS_OVERRIDE=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --aggregated)   AGGREGATED="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --metric)       METRIC="$2";       shift 2 ;;
        --alpha)        ALPHA="$2";        shift 2 ;;
        --top-n)        TOP_N_CD="$2";     shift 2 ;;
        --no-cd)        NO_CD=true;        shift ;;
        --all-metrics)  ALL_METRICS=true;  shift ;;
        --references)
            shift
            while [[ $# -gt 0 && "${1:0:2}" != "--" ]]; do
                REFS_OVERRIDE+=("$1"); shift
            done
            ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Override references if provided
if [[ ${#REFS_OVERRIDE[@]} -gt 0 ]]; then
    REFERENCES="${REFS_OVERRIDE[*]}"
fi

# ── Check inputs ──────────────────────────────────────────────────────────────
if [[ ! -f "$AGGREGATED" ]]; then
    echo "ERROR: aggregated CSV not found: $AGGREGATED"
    echo "       Run aggregate.py first, or pass --aggregated <path>."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

run_significance() {
    local metric="$1"
    local metric_out_dir="$OUTPUT_DIR"

    # If running multiple metrics, write each to a sub-directory
    if $ALL_METRICS; then
        metric_out_dir="$OUTPUT_DIR/significance_${metric// /_}"
        mkdir -p "$metric_out_dir"
    fi

    local cd_flag=""
    $NO_CD && cd_flag="--no-cd"

    echo ""
    echo "┌────────────────────────────────────────────────────────────┐"
    printf  "│  %-60s│\n" "Metric   : $metric"
    printf  "│  %-60s│\n" "Input    : $AGGREGATED"
    printf  "│  %-60s│\n" "Output   : $metric_out_dir"
    printf  "│  %-60s│\n" "Alpha    : $ALPHA"
    printf  "│  %-60s│\n" "Refs     : $REFERENCES"
    printf  "│  %-60s│\n" "Time     : $(timestamp)"
    echo    "└────────────────────────────────────────────────────────────┘"

    # shellcheck disable=SC2086
    $PYTHON -m survpfn.scripts.significance \
        --aggregated   "$AGGREGATED"    \
        --output-dir   "$metric_out_dir" \
        --metric       "$metric"         \
        --references   $REFERENCES       \
        --alpha        "$ALPHA"          \
        --top-n-cd     "$TOP_N_CD"       \
        $cd_flag
}

# ── Run ───────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "  SurvPFN Statistical Significance Tests"
echo "  aggregated : $AGGREGATED"
echo "  output_dir : $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════════"

if $ALL_METRICS; then
    for m in "C-index" "IBS" "AUC"; do
        run_significance "$m"
    done
else
    run_significance "$METRIC"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Done.  Results written to: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════════"
