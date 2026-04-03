"""
scripts/latex_table.py — Generate publication-ready LaTeX tables from SurvPFN benchmark results.

Reads ``aggregated.csv`` (produced by aggregate.py / xai pipeline) and produces
two tables:

1. **C-index table** — Models × Datasets, cells = mean ± std.
   Bold = best per column, underline = second-best, gray ±std subscript.

2. **IBS calibration table** — same layout, lower-is-better metric.

Table structure follows the same format as ``ehr_xai/performance/latex/format_result_l.py``:
  * Three-level header: Group → Dataset → Metric
  * Model groups (rows): Classical, Tree, Deep DL, FM Frozen, FM Joint, Zero-shot
  * Best/second highlighted per column
  * ``\\ensuremath{mean_{\\textcolor{gray}{\\pm std}}}`` cell format

CLI
---
    uv run python -m survpfn.scripts.latex_table
    uv run python -m survpfn.scripts.latex_table \\
        --aggregated results/benchmark/aggregated.csv \\
        --output-dir results/benchmark \\
        --metrics "C-index" "IBS" \\
        --datasets SUPPORT2 METABRIC GBSG WHAS500 VETERANS FLCHAIN
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model metadata: display names and grouping
# ---------------------------------------------------------------------------

#: (internal_name → display_name)
MODEL_DISPLAY: dict[str, str] = {
    # Classical
    "cox":            "Cox PH",
    "km":             "Kaplan-Meier",
    # Tree
    "rsf":            "RSF",
    "gbsa":           "GBSA",
    # Deep DL
    "deepsurv":       "DeepSurv",
    "mtlr":           "MTLR",
    "pchazard":       "PCHazard",
    "deephit_single": "DeepHit",
    "survtrace":      "SurvTRACE",
    "soden":          "SODEN",
    # TabPFN frozen embedding
    "tabpfn_embedding_cox":      "TabPFN+Cox",
    "tabpfn_embedding_deephit":  "TabPFN+DeepHit",
    "tabpfn_embedding_pchazard": "TabPFN+PCHazard",
    "tabpfn_embedding_mtlr":     "TabPFN+MTLR",
    # TabDPT frozen embedding
    "tabdpt_embedding_cox":      "TabDPT+Cox",
    "tabdpt_embedding_deephit":  "TabDPT+DeepHit",
    "tabdpt_embedding_pchazard": "TabDPT+PCHazard",
    "tabdpt_embedding_mtlr":     "TabDPT+MTLR",
    # TabICL frozen embedding
    "tabicl_embedding_cox":      "TabICL+Cox",
    "tabicl_embedding_deephit":  "TabICL+DeepHit",
    "tabicl_embedding_pchazard": "TabICL+PCHazard",
    "tabicl_embedding_mtlr":     "TabICL+MTLR",
    # TabPFN jointly trained
    "tabpfn_cox":      "TabPFN-Joint+Cox",
    "tabpfn_deephit":  "TabPFN-Joint+DeepHit",
    "tabpfn_pchazard": "TabPFN-Joint+PCHazard",
    "tabpfn_mtlr":     "TabPFN-Joint+MTLR",
    # TabDPT jointly trained
    "tabdpt_cox":      "TabDPT-Joint+Cox",
    "tabdpt_deephit":  "TabDPT-Joint+DeepHit",
    "tabdpt_pchazard": "TabDPT-Joint+PCHazard",
    "tabdpt_mtlr":     "TabDPT-Joint+MTLR",
    # TabICL jointly trained
    "tabicl_cox":      "TabICL-Joint+Cox",
    "tabicl_deephit":  "TabICL-Joint+DeepHit",
    "tabicl_pchazard": "TabICL-Joint+PCHazard",
    "tabicl_mtlr":     "TabICL-Joint+MTLR",
    # Zero-shot
    "tabpfn_zeroshot":         "TabPFN ZS",
    "tabdpt_zeroshot":         "TabDPT ZS",
    "tabicl_zeroshot":         "TabICL ZS",
    "tabpfn_zeroshot_perbin":  "TabPFN ZS-PB",
    "tabdpt_zeroshot_perbin":  "TabDPT ZS-PB",
    "tabicl_zeroshot_perbin":  "TabICL ZS-PB",
    # BetaSurv
    "beta_surv":   "BetaSurv",
}

#: Ordered model groups for row layout
MODEL_GROUPS: dict[str, list[str]] = {
    "Classical": ["cox", "km"],
    "Tree Ensemble": ["rsf", "gbsa"],
    "Deep Survival": ["deepsurv", "mtlr", "pchazard", "deephit_single", "survtrace", "soden"],
    "FM Frozen Embedding": [
        "tabpfn_embedding_cox", "tabpfn_embedding_deephit",
        "tabpfn_embedding_pchazard", "tabpfn_embedding_mtlr",
        "tabdpt_embedding_cox", "tabdpt_embedding_deephit",
        "tabdpt_embedding_pchazard", "tabdpt_embedding_mtlr",
        "tabicl_embedding_cox", "tabicl_embedding_deephit",
        "tabicl_embedding_pchazard", "tabicl_embedding_mtlr",
    ],
    "FM Jointly Trained": [
        "tabpfn_cox", "tabpfn_deephit", "tabpfn_pchazard", "tabpfn_mtlr",
        "tabdpt_cox", "tabdpt_deephit", "tabdpt_pchazard", "tabdpt_mtlr",
        "tabicl_cox", "tabicl_deephit", "tabicl_pchazard", "tabicl_mtlr",
    ],
    "Zero-Shot ICL": [
        "tabpfn_zeroshot", "tabdpt_zeroshot", "tabicl_zeroshot",
        "tabpfn_zeroshot_perbin", "tabdpt_zeroshot_perbin", "tabicl_zeroshot_perbin",
        "beta_surv",
    ],
}

#: Short dataset names for column headers (full name → short)
DATASET_SHORT: dict[str, str] = {
    "SUPPORT2":  "SUP2",
    "METABRIC":  "METAB",
    "GBSG":      "GBSG",
    "WHAS500":   "WHAS",
    "VETERANS":  "VET",
    "FLCHAIN":   "FLC",
}

# Public dataset group order
PUBLIC_DATASETS = ["SUPPORT2", "METABRIC", "GBSG", "WHAS500", "VETERANS", "FLCHAIN"]


# ---------------------------------------------------------------------------
# Metric metadata: display names
# ---------------------------------------------------------------------------

#: (internal_name → display_name)
METRIC_DISPLAY: dict[str, str] = {
    "C-index": "Concordance Index (C-index)",
    "IBS":     "Integrated Brier Score",
    "D-cal":   "D-calibration p-value",
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_aggregated(csv_path: str, metric: str) -> pd.DataFrame:
    """Load fold-level CSV; compute mean and std across folds per (Dataset, Model)."""
    df = pd.read_csv(csv_path)

    required = {"Dataset", "Model", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columns missing in {csv_path}: {missing}")

    df = df.dropna(subset=[metric])

    summary = (
        df.groupby(["Dataset", "Model"])[metric]
        .agg(mean="mean", std="std", n_folds="count")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def pivot_for_table(
    summary: pd.DataFrame,
    datasets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (mean_pivot, std_pivot) DataFrames with Model as index."""
    mean_p = summary.pivot(index="Model", columns="Dataset", values="mean")
    std_p  = summary.pivot(index="Model", columns="Dataset", values="std")

    # Keep only requested datasets that are present
    ds_present = [d for d in datasets if d in mean_p.columns]
    if not ds_present:
        warnings.warn("None of the requested datasets found in aggregated CSV.")
        ds_present = list(mean_p.columns)

    return mean_p[ds_present], std_p[ds_present]


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------

def _parse_val(v) -> float:
    if isinstance(v, (float, int)) and not np.isnan(v):
        return float(v)
    return np.nan


def format_cell(
    mean: float,
    std: float | None = None,
    style: str | None = None,        # "best" | "second" | None
    lower_is_better: bool = False,
) -> str:
    """Format a single cell as ``\\ensuremath{mean_{\\textcolor{gray}{\\pm std}}}``."""
    if np.isnan(mean):
        return "—"

    mean_str = f"{mean:.3f}"
    if style == "best":
        mean_str = f"\\textbf{{{mean_str}}}"
    elif style == "second":
        mean_str = f"\\underline{{{mean_str}}}"

    if std is None or np.isnan(std) or std == 0.0:
        return f"\\ensuremath{{{mean_str}}}"
    return f"\\ensuremath{{{mean_str}_{{\\textcolor{{gray}}{{\\pm {std:.3f}}}}}}}"


def compute_ranks(
    mean_pivot: pd.DataFrame,
    lower_is_better: bool = False,
) -> dict[str, dict[str, str]]:
    """For each column (dataset), find best and second-best models.

    Returns: ranks[dataset][model] = "best" | "second" | None
    """
    ranks: dict[str, dict[str, str]] = {}
    for col in mean_pivot.columns:
        col_vals = mean_pivot[col].dropna()
        sorted_vals = col_vals.sort_values(ascending=lower_is_better)
        ranks[col] = {}
        if len(sorted_vals) > 0:
            ranks[col][sorted_vals.index[0]] = "best"
        if len(sorted_vals) > 1:
            ranks[col][sorted_vals.index[1]] = "second"
    return ranks


# ---------------------------------------------------------------------------
# LaTeX table builder
# ---------------------------------------------------------------------------

def build_latex_table(
    mean_pivot: pd.DataFrame,
    std_pivot: pd.DataFrame,
    datasets: list[str],
    metric: str,
    lower_is_better: bool = False,
    dataset_groups: dict[str, list[str]] | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Render a complete LaTeX table*.

    Parameters
    ----------
    mean_pivot : (Models × Datasets) mean values.
    std_pivot  : (Models × Datasets) std values.
    datasets   : ordered list of dataset names to include.
    metric     : metric name (for header).
    lower_is_better : if True, sort ascending for best/second ranking.
    dataset_groups  : optional grouping of datasets into supercolumns.
                      e.g. {"Public": ["SUPPORT2", "METABRIC", ...], "SurvSet": [...]}
    caption, label  : LaTeX caption / label strings.
    """
    ds_present = [d for d in datasets if d in mean_pivot.columns]
    n_ds = len(ds_present)

    if n_ds == 0:
        return "% No data found for the requested datasets."

    # Short names for column headers
    col_headers = [DATASET_SHORT.get(d, d) for d in ds_present]

    # Compute best/second per column
    ranks = compute_ranks(mean_pivot[ds_present], lower_is_better=lower_is_better)

    # Ordered list of model rows (preserve group order; skip missing)
    all_models_present = set(mean_pivot.index)
    ordered_models: list[str] = []
    ordered_groups: list[tuple[str, list[str]]] = []
    for grp, mlist in MODEL_GROUPS.items():
        grp_models = [m for m in mlist if m in all_models_present]
        if grp_models:
            ordered_groups.append((grp, grp_models))
            ordered_models.extend(grp_models)
    # Any model not in a group → "Other"
    extra = [m for m in sorted(all_models_present) if m not in ordered_models]
    if extra:
        ordered_groups.append(("Other", extra))
        ordered_models.extend(extra)

    # Columns spec
    col_spec = "l" + "c" * n_ds
    n_total_cols = 1 + n_ds

    arrow = "$\\downarrow$" if lower_is_better else "$\\uparrow$"

    caption = caption or (
        f"\\textbf{{{METRIC_DISPLAY.get(metric, metric)}}} results (mean~$\\pm$~std across 5 folds). "
        "\\textbf{Bold} = best, \\underline{underline} = second-best per dataset."
    )
    label = label or f"tab:survpfn:{metric.lower().replace('-', '_')}"

    lines: list[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\renewcommand{\\arraystretch}{0.90}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # ── Header ────────────────────────────────────────────────────────────────
    if dataset_groups:
        # Row 1: Dataset group spans
        hdr1 = "\\multirow{2}{*}{\\textbf{Model}}"
        col_cursor = 2
        cmidrules = []
        for grp_name, grp_ds in dataset_groups.items():
            grp_ds_present = [d for d in grp_ds if d in ds_present]
            nc = len(grp_ds_present)
            if nc == 0:
                continue
            hdr1 += f" & \\multicolumn{{{nc}}}{{c}}{{\\textbf{{{grp_name}}}}}"
            cmidrules.append(f"\\cmidrule(lr){{{col_cursor}-{col_cursor + nc - 1}}}")
            col_cursor += nc
        lines.append(hdr1 + " \\\\")
        lines.append(" ".join(cmidrules))

        # Row 2: Short dataset names + arrow
        hdr2_parts = []
        for grp_name, grp_ds in dataset_groups.items():
            for d in grp_ds:
                if d in ds_present:
                    hdr2_parts.append(f"{DATASET_SHORT.get(d, d)} {arrow}")
        lines.append(" & " + " & ".join(hdr2_parts) + " \\\\")
    else:
        # Single-row header
        metric_hdrs = " & ".join(f"\\textbf{{{h}}} {arrow}" for h in col_headers)
        lines.append(f"\\textbf{{Model}} & {metric_hdrs} \\\\")

    lines.append("\\midrule")

    # ── Rows (by model group) ─────────────────────────────────────────────────
    for gi, (grp_name, grp_models) in enumerate(ordered_groups):
        # Group header row
        lines.append(
            f"\\multicolumn{{{n_total_cols}}}{{l}}"
            f"{{\\textit{{\\small {grp_name}}}}} \\\\"
        )
        lines.append("\\midrule")

        for m in grp_models:
            display = MODEL_DISPLAY.get(m, m)
            cells: list[str] = [display]
            for d in ds_present:
                mean = _parse_val(
                    mean_pivot.loc[m, d] if m in mean_pivot.index and d in mean_pivot.columns
                    else np.nan
                )
                std = _parse_val(
                    std_pivot.loc[m, d] if m in std_pivot.index and d in std_pivot.columns
                    else np.nan
                )
                style = ranks.get(d, {}).get(m, None)
                cells.append(format_cell(mean, std, style, lower_is_better))
            lines.append(" & ".join(cells) + " \\\\")

        # Separator between groups (except last)
        if gi < len(ordered_groups) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact summary table (selected models only, for the main paper)
# ---------------------------------------------------------------------------

def build_summary_table(
    mean_pivot: pd.DataFrame,
    std_pivot: pd.DataFrame,
    datasets: list[str],
    metric: str,
    selected_models: list[str] | None = None,
    lower_is_better: bool = False,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """A compact version of the full table with only the most important models.

    The paper-ready Table 2 / Table 3 should be compact.  This selects:
    - Best classical baseline (cox)
    - Best tree baseline (rsf)
    - Best deep baseline (deephit_single)
    - All FM frozen embedding best-head variants (DeepHit per backbone)
    - All FM jointly trained best-head variants
    - Zero-shot ICL
    """
    if selected_models is None:
        selected_models = [
            "cox", "rsf", "deephit_single",
            "tabpfn_embedding_deephit", "tabdpt_embedding_deephit", "tabicl_embedding_deephit",
            "tabpfn_deephit", "tabdpt_deephit", "tabicl_deephit",
            "tabpfn_zeroshot", "tabdpt_zeroshot", "tabicl_zeroshot",
            "soden", "beta_surv",
        ]

    present = [m for m in selected_models if m in mean_pivot.index]
    ds_present = [d for d in datasets if d in mean_pivot.columns]

    if not present or not ds_present:
        return "% No data for summary table."

    col_headers = [DATASET_SHORT.get(d, d) for d in ds_present]
    ranks = compute_ranks(mean_pivot[ds_present].loc[present], lower_is_better=lower_is_better)

    col_spec = "l" + "c" * len(ds_present)
    n_total = 1 + len(ds_present)
    arrow = "$\\downarrow$" if lower_is_better else "$\\uparrow$"

    caption = caption or (
        f"Summary {METRIC_DISPLAY.get(metric, metric)} results (mean~$\\pm$~std, 5-fold CV). "
        "\\textbf{Bold} = best, \\underline{underline} = second-best."
    )
    label = label or f"tab:survpfn:summary:{metric.lower().replace('-', '_')}"

    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\columnwidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    metric_hdrs = " & ".join(f"\\textbf{{{h}}} {arrow}" for h in col_headers)
    lines.append(f"\\textbf{{Model}} & {metric_hdrs} \\\\")
    lines.append("\\midrule")

    # Group rows with thin separators
    group_ranges = [
        ("Baselines",           ["cox", "rsf", "deephit_single"]),
        ("FM Frozen",           ["tabpfn_embedding_deephit", "tabdpt_embedding_deephit",
                                 "tabicl_embedding_deephit"]),
        ("FM Joint",            ["tabpfn_deephit", "tabdpt_deephit", "tabicl_deephit"]),
        ("Zero-Shot / Other",   ["tabpfn_zeroshot", "tabdpt_zeroshot", "tabicl_zeroshot",
                                 "soden", "beta_surv"]),
    ]

    for gi, (grp_name, grp_models) in enumerate(group_ranges):
        grp_present = [m for m in grp_models if m in present]
        if not grp_present:
            continue
        lines.append(
            f"\\multicolumn{{{n_total}}}{{l}}"
            f"{{\\textit{{\\footnotesize {grp_name}}}}} \\\\"
        )
        for m in grp_present:
            display = MODEL_DISPLAY.get(m, m)
            cells: list[str] = [display]
            for d in ds_present:
                mean = _parse_val(
                    mean_pivot.loc[m, d] if m in mean_pivot.index and d in mean_pivot.columns
                    else np.nan
                )
                std = _parse_val(
                    std_pivot.loc[m, d] if m in std_pivot.index and d in std_pivot.columns
                    else np.nan
                )
                style = ranks.get(d, {}).get(m, None)
                cells.append(format_cell(mean, std, style, lower_is_better))
            lines.append(" & ".join(cells) + " \\\\")
        if gi < len(group_ranges) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from SurvPFN benchmark aggregated results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--aggregated", default="results/benchmark/aggregated.csv",
        help="Path to aggregated.csv.",
    )
    parser.add_argument(
        "--output-dir", default="results/benchmark",
        help="Directory where .tex files are written.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["C-index", "IBS"],
        help="Metrics to tabulate (one table per metric).",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=PUBLIC_DATASETS,
        help="Ordered list of datasets for columns.",
    )
    parser.add_argument(
        "--lower-is-better-metrics", nargs="+", default=["IBS", "D-cal"],
        help="Metrics where lower is better (affects bold/underline).",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Also generate compact summary tables (selected models only).",
    )
    parser.add_argument(
        "--group-datasets", action="store_true",
        help="Add dataset group header (Public Benchmarks).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_groups: dict | None = None
    if args.group_datasets:
        pub = [d for d in args.datasets if not d.startswith("SS_")]
        ss  = [d for d in args.datasets if d.startswith("SS_")]
        dataset_groups = {}
        if pub:
            dataset_groups["Public Benchmarks"] = pub
        if ss:
            dataset_groups["SurvSet"] = ss
        if not dataset_groups:
            dataset_groups = None

    for metric in args.metrics:
        # Map "Integrated Brier Score" to "IBS" if that's what's in the CSV
        col_name = metric
        if col_name == "Integrated Brier Score":
            col_name = "IBS"
        elif col_name not in ["C-index", "IBS", "D-cal"] and "IBS" in metric:
             col_name = "IBS"

        print(f"\n[{metric}] Loading {args.aggregated} … (using CSV column '{col_name}')")
        try:
            summary = load_aggregated(args.aggregated, col_name)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        mean_pivot, std_pivot = pivot_for_table(summary, args.datasets)
        lower = col_name in args.lower_is_better_metrics or metric in args.lower_is_better_metrics

        # Full table
        print(f"  Building full table for {metric} ({len(mean_pivot.index)} models, "
              f"{len([d for d in args.datasets if d in mean_pivot.columns])} datasets) …")
        tex = build_latex_table(
            mean_pivot, std_pivot,
            datasets=args.datasets,
            metric=metric,
            lower_is_better=lower,
            dataset_groups=dataset_groups,
        )
        out_path = out_dir / f"table_{metric.replace('-', '_').lower()}_full.tex"
        out_path.write_text(tex)
        print(f"  Saved: {out_path}")
        print("\n" + tex[:600] + ("…" if len(tex) > 600 else "") + "\n")

        # Summary table
        if args.summary:
            tex_s = build_summary_table(
                mean_pivot, std_pivot,
                datasets=args.datasets,
                metric=metric,
                lower_is_better=lower,
            )
            out_s = out_dir / f"table_{metric.replace('-', '_').lower()}_summary.tex"
            out_s.write_text(tex_s)
            print(f"  Summary saved: {out_s}")

    print(f"\nAll tables written to: {out_dir}")


if __name__ == "__main__":
    main()
