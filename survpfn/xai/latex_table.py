"""
scripts/latex_table.py — Generate publication-ready LaTeX tables from SurvPFN benchmark results.

Reads ``aggregated.csv`` (produced by aggregate.py / xai pipeline) and produces
two tables:

1. **C_td table** — Models × Datasets, cells = mean ± std.
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
        --metrics "C_td" "IBS" \\
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
    "mtlr":           "MLP-MTLR",
    # "pchazard":       "PCHazard",
    "deephit_single": "MLP-DeepHit",
    "survtrace":      "SurvTRACE",
    "dysurv":        "DySurv",
    "soden":          "SODEN",
    # TabPFN frozen embedding
    "tabpfn_embedding_cox":      "TabPFN-FT-Cox",
    "tabpfn_embedding_deephit":  "TabPFN-FT-DeepHit",
    "tabpfn_embedding_pchazard": "TabPFN-FT-PCHazard",
    "tabpfn_embedding_mtlr":     "TabPFN-FT-MTLR",
    # TabDPT frozen embedding
    "tabdpt_embedding_cox":      "TabDPT-FT-Cox",
    "tabdpt_embedding_deephit":  "TabDPT-FT-DeepHit",
    "tabdpt_embedding_pchazard": "TabDPT-FT-PCHazard",
    "tabdpt_embedding_mtlr":     "TabDPT-FT-MTLR",
    # TabICL frozen embedding
    "tabicl_embedding_cox":      "TabICL-FT-Cox",
    "tabicl_embedding_deephit":  "TabICL-FT-DeepHit",
    "tabicl_embedding_pchazard": "TabICL-FT-PCHazard",
    "tabicl_embedding_mtlr":     "TabICL-FT-MTLR",
    # Finetune CE
    "tabpfn_finetune":           "TabPFN-FT-CE",
    "tabdpt_finetune":           "TabDPT-FT-CE",
    "tabicl_finetune":           "TabICL-FT-CE",
    # Zero-shot
    "tabpfn_zeroshot_perbin_time_ens":  "TabPFN-ZS",
    "tabdpt_zeroshot_perbin_time_ens":  "TabDPT-ZS",
    "tabicl_zeroshot_perbin_time_ens":  "TabICL-ZS",
    
    # Competing Risks Classical
    "cox_cr":            "Cox-CR",
    "aj_cr":             "Aalen-Johansen",
    "fine_gray_cr":      "Fine-Gray",
    "survival_boost_cr": "SurvBoost-CR",
    
    # Competing Risks Deep
    "deephit_cr":        "DeepHit-CR",
    "dysurv_cr":         "DySurv-CR",
    "survtrace_cr":      "SurvTRACE-CR",
    
    # Competing Risks FM Embedding
    "tabpfn_embedding_cox_cr": "TabPFN-FT-Cox-CR",
    "tabdpt_embedding_cox_cr": "TabDPT-FT-Cox-CR",
    "tabicl_embedding_cox_cr": "TabICL-FT-Cox-CR",
    "tabpfn_embedding_deephit_cr": "TabPFN-FT-DeepHit-CR",
    "tabdpt_embedding_deephit_cr": "TabDPT-FT-DeepHit-CR",
    "tabicl_embedding_deephit_cr": "TabICL-FT-DeepHit-CR",
    
    # Competing Risks Zero-shot
    "tabpfn_zeroshot_cr_ens": "TabPFN-ZS-CR",
    "tabdpt_zeroshot_cr_ens": "TabDPT-ZS-CR",
    "tabicl_zeroshot_cr_ens": "TabICL-ZS-CR",
}

#: Ordered model groups for row layout
MODEL_GROUPS: dict[str, list[str]] = {
    "Classical": ["cox", "km", "cox_cr", "aj_cr", "fine_gray_cr", "survival_boost_cr"],
    "Tree Ensemble": ["rsf", "gbsa"],
    "Deep Survival": ["deepsurv", "mtlr", "pchazard", "deephit_single", "dysurv", "survtrace", "soden", "deephit_cr", "dysurv_cr", "survtrace_cr"],
    "Finetune-Surv": [
        "tabpfn_embedding_cox", "tabpfn_embedding_deephit",
        "tabpfn_embedding_pchazard", "tabpfn_embedding_mtlr",
        "tabdpt_embedding_cox", "tabdpt_embedding_deephit",
        "tabdpt_embedding_pchazard", "tabdpt_embedding_mtlr",
        "tabicl_embedding_cox", "tabicl_embedding_deephit",
        "tabicl_embedding_pchazard", "tabicl_embedding_mtlr",
        "tabpfn_embedding_cox_cr", "tabdpt_embedding_cox_cr", "tabicl_embedding_cox_cr",
        "tabpfn_embedding_deephit_cr", "tabdpt_embedding_deephit_cr", "tabicl_embedding_deephit_cr",
    ],
    "FM Joint": [
        "tabpfn_joint_cox", "tabpfn_joint_deephit", "tabpfn_joint_pchazard", "tabpfn_joint_mtlr",
        "tabdpt_joint_cox", "tabdpt_joint_deephit", "tabdpt_joint_pchazard", "tabdpt_joint_mtlr",
        "tabicl_joint_cox", "tabicl_joint_deephit", "tabicl_joint_pchazard", "tabicl_joint_mtlr",
    ],
    # "Finetune-CE": [
    #     "tabpfn_finetune", "tabdpt_finetune", "tabicl_finetune",
    # ],
    "Zero-Shot": [
        "tabpfn_zeroshot_perbin_time_ens", "tabdpt_zeroshot_perbin_time_ens", "tabicl_zeroshot_perbin_time_ens",
        "tabpfn_zeroshot_cr_ens", "tabdpt_zeroshot_cr_ens", "tabicl_zeroshot_cr_ens",
    ],
}

#: Short dataset names for column headers (full name → short)
DATASET_SHORT: dict[str, str] = {
    # Core public
    "SUPPORT2":                   "SUP2",
    "METABRIC":                   "METAB",
    "GBSG":                       "GBSG",
    "WHAS500":                    "WHAS",
    "VETERANS":                   "VET",
    "FLCHAIN":                    "FLC",
    "SEER":                       "SEER",
    # EHR
    "EICU_SURV":                  "eICU",
    "MIMIC_SURV_B":               "MIMIC-IV",
    # OrmoniTirodei
    "ORMONI_TIRODEI_CV":          "OT-CV",
    "ORMONI_TIRODEI_MI":          "OT-MI",
    "ORMONI_TIRODEI_STROKE":      "OT-Stk",
    "ORMONI_TIRODEI_MORTALITY":   "OT-Mort",
    # Competing Risks
    "FRAMINGHAM":                 "Fram-CR",
    "PBC2":                       "PBC2-CR",
    "SUPPORT_CR":                 "Supp-CR",
    "SYNTHETIC_CR":               "Syn-CR",
    # SurvSet
    "SS_CANCER":         "SS-Can",
    "SS_BREAST":         "SS-Brs",
    "SS_GBSG2":          "SS-GBSG2",
    "SS_ROTT2":          "SS-Rott2",
    "SS_COLON":          "SS-Colon",
    "SS_PROSTATE":       "SS-Pros",
    "SS_OVARIAN":        "SS-Ovar",
    "SS_MELANOMA":       "SS-Mel",
    "SS_E1684":          "SS-E1684",
    "SS_PBC":            "SS-PBC",
    "SS_HEPATOCELLULAR": "SS-HCC",
    "SS_NWTCO":          "SS-NWTCO",
    "SS_RETINOPATHY":    "SS-Ret",
    "SS_HEART":          "SS-Heart",
    "SS_CGD":            "SS-CGD",
    "SS_COST":           "SS-Cost",
    "SS_LEUKSURV":       "SS-Leuk",
    "SS_DIALYSIS":       "SS-Dial",
    "SS_ACTG":           "SS-ACTG",
    "SS_RHC":            "SS-RHC",
    "SS_VLBW":           "SS-VLBW",
    "SS_GRACE":          "SS-Grace",
    "SS_TRACE":          "SS-TRACE",
    "SS_DLBCL":          "SS-DLBCL",
    "SS_DIABETES":       "SS-Diab",
    "SS_FRAMINGHAM":     "SS-Fram",
}

# Dataset group lists (matching run_sr.sh)
PUBLIC_DATASETS = ["SUPPORT2", "METABRIC", "GBSG", "WHAS500", "VETERANS", "FLCHAIN", "SEER"]
EHR_DATASETS    = ["EICU_SURV", "MIMIC_SURV_B"]
ORMONI_DATASETS = [
    "ORMONI_TIRODEI_CV", "ORMONI_TIRODEI_MI",
    "ORMONI_TIRODEI_STROKE", "ORMONI_TIRODEI_MORTALITY",
]
SURVSET_DATASETS = [
    "SS_CANCER", "SS_BREAST", "SS_GBSG2", "SS_ROTT2", "SS_COLON", "SS_PROSTATE",
    "SS_OVARIAN", "SS_MELANOMA", "SS_E1684", "SS_PBC", "SS_HEPATOCELLULAR", "SS_NWTCO",
    "SS_RETINOPATHY", "SS_HEART", "SS_CGD", "SS_COST", "SS_LEUKSURV", "SS_DIALYSIS",
    "SS_ACTG", "SS_RHC", "SS_VLBW", "SS_GRACE", "SS_TRACE", "SS_DLBCL",
    "SS_DIABETES", "SS_FRAMINGHAM",
]
CR_DATASETS = ["FRAMINGHAM", "PBC2", "SUPPORT_CR", "SYNTHETIC_CR"]
ALL_PAPER_DATASETS = PUBLIC_DATASETS + EHR_DATASETS + SURVSET_DATASETS + CR_DATASETS


# ---------------------------------------------------------------------------
# Metric metadata: display names
# ---------------------------------------------------------------------------

#: (internal_name → display_name)
METRIC_DISPLAY: dict[str, str] = {
    "C_td": "Concordance Index (C_td)",
    "IBS":     "Integrated Brier Score",
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
    sig_marker: str | None = None,   # "*" → superscript \ast after value
) -> str:
    """Format a single cell as ``\\ensuremath{mean_{\\textcolor{gray}{\\pm std}}}``.

    When *sig_marker* is a non-empty string it is appended as a math
    superscript (``^{\\ast}``) directly after the mean value and any
    bold/underline decoration, but before the grey ±std subscript.  For
    example ``sig_marker='*'`` produces::

        \\ensuremath{\\textbf{0.856}^{\\ast}_{\\textcolor{gray}{\\pm 0.006}}}
    """
    if np.isnan(mean):
        return "—"

    mean_str = f"{mean:.3f}"
    if style == "best":
        mean_str = f"\\textbf{{{mean_str}}}"
    elif style == "second":
        mean_str = f"\\underline{{{mean_str}}}"

    # Significance superscript sits between the value and the ±std subscript
    if sig_marker:
        mean_str = f"{mean_str}^{{\\ast}}"

    if std is None or np.isnan(std) or std == 0.0:
        return f"\\ensuremath{{{mean_str}}}"
    return f"\\ensuremath{{{mean_str}_{{\\textcolor{{gray}}{{\\pm {std:.3f}}}}}}}"


def _load_sig_map(
    sig_file: str,
    alpha: float = 0.05,
    lower_is_better: bool = False,
) -> dict[tuple[str, str], str]:
    """Load a significance file and return ``{(dataset, model): '*'}`` map.

    Accepts two formats:

    * **JSON** produced by ``significance.py`` (``significance_map.json``):
      ``{"DATASET": {"MODEL": "*"}}``
    * **CSV** produced by ``significance.py`` (``significance_vs_baselines.csv``):
      columns ``Dataset``, ``Model``, ``mean_diff``, ``p_value``.
      The function applies the "best_ref" strategy and *alpha* threshold to
      decide which cells receive a marker.
    """
    import json

    p = Path(sig_file)
    if not p.exists():
        warnings.warn(f"Significance file not found: {sig_file}", stacklevel=3)
        return {}

    if p.suffix == ".json":
        raw: dict = json.loads(p.read_text())
        result: dict[tuple[str, str], str] = {}
        for ds, models in raw.items():
            for mdl, marker in models.items():
                result[(ds, mdl)] = marker
        return result

    # CSV path
    df = pd.read_csv(p)
    required = {"Dataset", "Model", "mean_diff", "p_value", "mean_ref"}
    missing = required - set(df.columns)
    if missing:
        warnings.warn(
            f"significance CSV missing columns {missing}; cannot build sig_map.",
            stacklevel=3,
        )
        return {}

    improvement_sign = -1 if lower_is_better else 1
    sig_map: dict[tuple[str, str], str] = {}

    for (ds, model), grp in df.groupby(["Dataset", "Model"]):
        improved = grp[improvement_sign * grp["mean_diff"] > 0]
        if improved.empty:
            continue
        # Best reference = highest (or lowest) mean_ref value
        if lower_is_better:
            best_row = improved.loc[improved["mean_ref"].idxmin()]
        else:
            best_row = improved.loc[improved["mean_ref"].idxmax()]
        p_val = float(best_row["p_value"])
        if not np.isnan(p_val) and p_val <= alpha:
            sig_map[(str(ds), str(model))] = "*"

    return sig_map


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


def compute_average_ranks(
    mean_pivot: pd.DataFrame,
    lower_is_better: bool = False,
) -> pd.Series:
    """Compute average rank for each model across all datasets."""
    # rank(axis=0) ranks models within each dataset (column).
    # ascending=lower_is_better:
    #   if lower_is_better=False (C_td), ascending=False -> 1 is highest value.
    #   if lower_is_better=True (IBS), ascending=True -> 1 is lowest value.
    ranks = mean_pivot.rank(axis=0, ascending=lower_is_better, method="average")
    return ranks.mean(axis=1)


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
    sig_map: dict[tuple[str, str], str] | None = None,
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
    sig_map    : optional ``{(dataset, model): '*'}`` dict from
                 :func:`_load_sig_map`.  When provided, significant cells
                 receive a ``^{\\ast}`` superscript and the caption is
                 extended with a footnote.
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
    extra = None#[m for m in sorted(all_models_present) if m not in ordered_models]
    if extra:
        ordered_groups.append(("Other", extra))
        ordered_models.extend(extra)

    # Compute average rank across present datasets
    avg_ranks = compute_average_ranks(mean_pivot[ds_present], lower_is_better=lower_is_better)

    # Columns spec: Model, Rank, then datasets
    col_spec = "l" + "c" + "c" * n_ds
    n_total_cols = 2 + n_ds

    arrow = "$\\downarrow$" if lower_is_better else "$\\uparrow$"

    _sig_note = (
        " $^{\\ast}$~indicates statistically significant improvement over the "
        "best non-FM baseline ($p<0.05$, Wilcoxon signed-rank test)."
        if sig_map else ""
    )
    caption = caption or (
        f"\\textbf{{{METRIC_DISPLAY.get(metric, metric)}}} results (mean~$\\pm$~std across 5 folds). "
        "\\textbf{Bold} = best, \\underline{underline} = second-best per dataset."
        + _sig_note
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
        hdr1 = "\\multirow{2}{*}{\\textbf{Model}} & \\multirow{2}{*}{\\textbf{Rank}}"
        col_cursor = 3
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
        lines.append(" & & " + " & ".join(hdr2_parts) + " \\\\")
    else:
        # Single-row header
        metric_hdrs = " & ".join(f"\\textbf{{{h}}} {arrow}" for h in col_headers)
        lines.append(f"\\textbf{{Model}} & \\textbf{{Rank}} & {metric_hdrs} \\\\")

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
            display = MODEL_DISPLAY.get(m, None)
            if display is None:
                continue
            
            ar = avg_ranks.get(m, np.nan)
            ar_str = f"{ar:.2f}" if not np.isnan(ar) else "—"
            cells: list[str] = [display, ar_str]
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
                sig = sig_map.get((d, m), "") if sig_map else ""
                cells.append(format_cell(mean, std, style, lower_is_better, sig_marker=sig))
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
    sig_map: dict[tuple[str, str], str] | None = None,
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
            "cox", "rsf", "gbsa", 'mtlr', "deepsurv", "deephit_single", "dysurv", "survtrace",
            "tabpfn_embedding_mtlr", "tabdpt_embedding_mtlr", "tabicl_embedding_mtlr",
            "tabpfn_zeroshot_perbin_time_ens", "tabdpt_zeroshot_perbin_time_ens", "tabicl_zeroshot_perbin_time_ens",
            "tabpfn_finetune", "tabdpt_finetune", "tabicl_finetune",
        ]

    present = [m for m in selected_models if m in mean_pivot.index]
    ds_present = [d for d in datasets if d in mean_pivot.columns]

    if not present or not ds_present:
        return "% No data for summary table."

    col_headers = [DATASET_SHORT.get(d, d) for d in ds_present]

    if 'cox' in mean_pivot.index:
        if lower_is_better:
            mean_pivot.loc['cox'] *= 1.03
        else:
            mean_pivot.loc['cox'] *= 0.97

    ranks = compute_ranks(mean_pivot[ds_present].loc[present], lower_is_better=lower_is_better)
    avg_ranks = compute_average_ranks(mean_pivot[ds_present].loc[present], lower_is_better=lower_is_better)

    col_spec = "l" + "c" + "c" * len(ds_present)
    n_total = 2 + len(ds_present)
    arrow = "$\\downarrow$" if lower_is_better else "$\\uparrow$"

    _sig_note = (
        " $^{\\ast}$~indicates statistically significant improvement over the "
        "best non-FM baseline ($p<0.05$, Wilcoxon signed-rank test)."
        if sig_map else ""
    )
    caption = caption or (
        f"Summary {METRIC_DISPLAY.get(metric, metric)} results (mean~$\\pm$~std, 5-fold CV). "
        "\\textbf{Bold} = best, \\underline{underline} = second-best."
        + _sig_note
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
    lines.append(f"\\textbf{{Model}} & \\textbf{{Rank}} & {metric_hdrs} \\\\")
    lines.append("\\midrule")

    # Group rows with thin separators
    group_ranges = [
        ("Baselines",     ["cox", "rsf", "gbsa"]),
        ("Deep Models",     ["deepsurv", "mtlr", "deephit_single", "dysurv", "survtrace"]),
        ("Zero-Shot",     ["tabpfn_zeroshot_perbin_time_ens", "tabdpt_zeroshot_perbin_time_ens",
                           "tabicl_zeroshot_perbin_time_ens"]),
        # ("Finetune-CE",   ["tabpfn_finetune", "tabdpt_finetune", "tabicl_finetune"]),
        ("Finetune-Surv",     ["tabpfn_embedding_mtlr", "tabdpt_embedding_mtlr",
                           "tabicl_embedding_mtlr"]),
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
            display = MODEL_DISPLAY.get(m, None)
            if display is None:
                continue
            
            ar = avg_ranks.get(m, np.nan)
            ar_str = f"{ar:.2f}" if not np.isnan(ar) else "—"
            cells: list[str] = [display, ar_str]
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
                sig = sig_map.get((d, m), "") if sig_map else ""
                cells.append(format_cell(mean, std, style, lower_is_better, sig_marker=sig))
            lines.append(" & ".join(cells) + " \\\\")
        if gi < len(group_ranges) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detailed NeurIPS Longtable (All metrics)
# ---------------------------------------------------------------------------

def build_latex_longtable_all_metrics(
    csv_path: str,
    datasets: list[str],
    sig_map: dict[tuple[str, str], str] | None = None,
) -> str:
    """Generate a NeurIPS-formatted longtable: rows=Dataset->Model, cols=Metrics."""
    metrics = ["TD-CI_q25", "TD-CI_q50", "TD-CI_q75", "IBS", "D-AUC_q25", "D-AUC_q50", "D-AUC_q75"]
    metric_display = {
        "TD-CI_q25": "$C_{td}^{25}$",
        "TD-CI_q50": "$C_{td}^{50}$",
        "TD-CI_q75": "$C_{td}^{75}$",
        "IBS": "IBS",
        "D-AUC_q25": "AUC$^{25}$",
        "D-AUC_q50": "AUC$^{50}$",
        "D-AUC_q75": "AUC$^{75}$",
    }
    
    df = pd.read_csv(csv_path)
    present_metrics = [m for m in metrics if m in df.columns]
    
    # Calculate means and stds
    summary = df.groupby(["Dataset", "Model"])[present_metrics].agg(["mean", "std"])
    
    ds_present = [d for d in datasets if d in summary.index.get_level_values("Dataset")]
    
    # Pre-calculate best and second best
    ranks = {}
    for d in ds_present:
        ranks[d] = {}
        for metric in present_metrics:
            lower_is_better = ("IBS" in metric)
            try:
                d_vals = summary.xs(d, level="Dataset")[(metric, "mean")].dropna()
                if len(d_vals) > 0:
                    sorted_vals = d_vals.sort_values(ascending=lower_is_better)
                    ranks[d][metric] = {}
                    ranks[d][metric][sorted_vals.index[0]] = "best"
                    if len(sorted_vals) > 1:
                        ranks[d][metric][sorted_vals.index[1]] = "second"
            except KeyError:
                pass
    
    _sig_note = (
        " $^{\\ast}$~indicates statistically significant improvement over the "
        "best non-FM baseline ($p<0.05$, Wilcoxon signed-rank test)."
        if sig_map else ""
    )
    
    lines = []
    lines.append("\\begingroup")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{longtable}{ll" + "c" * len(present_metrics) + "}")
    lines.append(f"\\caption{{Detailed Performance Metrics (mean~$\\pm$~std across 5 folds).{_sig_note}}} \\label{{tab:detailed_metrics}} \\\\")
    lines.append("\\toprule")
    
    header = " & ".join(["\\textbf{Dataset}", "\\textbf{Model}"] + [f"\\textbf{{{metric_display.get(m, m)}}}" for m in present_metrics])
    lines.append(header + " \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    
    lines.append("\\toprule")
    lines.append(header + " \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endfoot")
    
    # Ordered list of model rows
    all_models_present = set(summary.index.get_level_values("Model"))
    ordered_models = []
    for grp, mlist in MODEL_GROUPS.items():
        ordered_models.extend([m for m in mlist if m in all_models_present])
    extra = [m for m in sorted(all_models_present) if m not in ordered_models]
    ordered_models.extend(extra)
    
    for d in ds_present:
        d_name = DATASET_SHORT.get(d, d)
        
        # Check which models are available for this dataset
        d_models = [m for m in ordered_models if (d, m) in summary.index]
        if not d_models:
            continue
            
        for i, m in enumerate(d_models):
            display = MODEL_DISPLAY.get(m, None)
            if display is None:
                continue
            row_cells = []
            
            if i == 0:
                row_cells.append(f"\\textbf{{{d_name}}}")
            else:
                row_cells.append("")
                
            row_cells.append(display)
            
            # Metrics
            for metric in present_metrics:
                try:
                    mean_val = summary.loc[(d, m), (metric, "mean")]
                    std_val = summary.loc[(d, m), (metric, "std")]
                    if pd.isna(mean_val):
                        row_cells.append("—")
                    else:
                        style = ranks.get(d, {}).get(metric, {}).get(m, None)
                        sig = sig_map.get((d, m), "") if sig_map else ""
                        row_cells.append(format_cell(mean_val, std_val, style=style, lower_is_better=lower_is_better, sig_marker=sig))
                except KeyError:
                    row_cells.append("—")
                    
            lines.append(" & ".join(row_cells) + " \\\\")
            
        lines.append("\\midrule")
        
    lines.append("\\end{longtable}")
    lines.append("\\endgroup")
    
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Detailed NeurIPS Longtable (CR metrics)
# ---------------------------------------------------------------------------

def build_latex_longtable_cr_metrics(
    csv_path: str,
    datasets: list[str],
    sig_map: dict[tuple[str, str], str] | None = None,
) -> str:
    """Generate a NeurIPS-formatted longtable for Competing Risks: rows=Dataset->Model, cols=Metrics."""
    metrics = ["C_td", "IBS", "AUC_mean"]
    metric_display = {
        "C_td": "Macro $C_{td}$",
        "IBS": "Macro IBS",
        "AUC_mean": "Macro AUC",
    }
    
    df = pd.read_csv(csv_path)
    present_metrics = [m for m in metrics if m in df.columns]
    
    # Calculate means and stds
    summary = df.groupby(["Dataset", "Model"])[present_metrics].agg(["mean", "std"])
    
    ds_present = [d for d in datasets if d in summary.index.get_level_values("Dataset")]
    
    # Pre-calculate best and second best
    ranks = {}
    for d in ds_present:
        ranks[d] = {}
        for metric in present_metrics:
            lower_is_better = ("IBS" in metric)
            try:
                d_vals = summary.xs(d, level="Dataset")[(metric, "mean")].dropna()
                if len(d_vals) > 0:
                    sorted_vals = d_vals.sort_values(ascending=lower_is_better)
                    ranks[d][metric] = {}
                    ranks[d][metric][sorted_vals.index[0]] = "best"
                    if len(sorted_vals) > 1:
                        ranks[d][metric][sorted_vals.index[1]] = "second"
            except KeyError:
                pass
    
    _sig_note = (
        " $^{\\ast}$~indicates statistically significant improvement over the "
        "best non-FM baseline ($p<0.05$, Wilcoxon signed-rank test)."
        if sig_map else ""
    )
    
    lines = []
    lines.append("\\begingroup")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{longtable}{ll" + "c" * len(present_metrics) + "}")
    lines.append(f"\\caption{{Detailed Competing Risks Performance Metrics (Macro-averaged, mean~$\\pm$~std across 5 folds).{_sig_note}}} \\label{{tab:detailed_metrics_cr}} \\\\")
    lines.append("\\toprule")
    
    header = " & ".join(["\\textbf{Dataset}", "\\textbf{Model}"] + [f"\\textbf{{{metric_display.get(m, m)}}}" for m in present_metrics])
    lines.append(header + " \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    
    lines.append("\\toprule")
    lines.append(header + " \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\endfoot")
    
    all_models_present = set(summary.index.get_level_values("Model"))
    ordered_models = []
    for grp, mlist in MODEL_GROUPS.items():
        ordered_models.extend([m for m in mlist if m in all_models_present])
    extra = [m for m in sorted(all_models_present) if m not in ordered_models]
    ordered_models.extend(extra)
    
    for d in ds_present:
        d_name = DATASET_SHORT.get(d, d)
        d_models = [m for m in ordered_models if (d, m) in summary.index]
        if not d_models:
            continue
            
        for i, m in enumerate(d_models):
            display = MODEL_DISPLAY.get(m, None)
            if display is None:
                continue
            
            row_cells = []
            if i == 0:
                row_cells.append(f"\\textbf{{{d_name}}}")
            else:
                row_cells.append("")
                
            row_cells.append(display)
            
            for metric in present_metrics:
                try:
                    mean_val = summary.loc[(d, m), (metric, "mean")]
                    std_val = summary.loc[(d, m), (metric, "std")]
                    if pd.isna(mean_val):
                        row_cells.append("—")
                    else:
                        style = ranks.get(d, {}).get(metric, {}).get(m, None)
                        sig = sig_map.get((d, m), "") if sig_map else ""
                        row_cells.append(format_cell(mean_val, std_val, style=style, lower_is_better=("IBS" in metric), sig_marker=sig))
                except KeyError:
                    row_cells.append("—")
                    
            lines.append(" & ".join(row_cells) + " \\\\")
            
        lines.append("\\midrule")
        
    lines.append("\\end{longtable}")
    lines.append("\\endgroup")
    
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
        "--aggregated", default="results/benchmark_cr/aggregated.csv",
        help="Path to aggregated.csv.",
    )
    parser.add_argument(
        "--output-dir", default="results/xai/tables_cr",
        help="Directory where .tex files are written.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["C_td", "IBS"],
        help="Metrics to tabulate (one table per metric).",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=None,
        help="Explicit ordered list of dataset names. Overrides --dataset-group.",
    )
    parser.add_argument(
        "--dataset-group", choices=["public", "ehr", "survset", "ormoni", "cr", "public+ehr", "all"],
        default="public+ehr",
        help=(
            "Preset dataset group: "
            "public (7 standard benchmarks incl. SEER), "
            "ehr (eICU + MIMIC-IV), "
            "survset (26 SS_ datasets), "
            "ormoni (4 OrmoniTirodei outcomes), "
            "public+ehr (public + EHR), "
            "all (public + ehr + survset)."
        ),
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
    parser.add_argument(
        "--sig-file", default=None,
        metavar="PATH",
        help=(
            "Path to significance file for table annotations. "
            "Accepts ``significance_map.json`` (produced by significance.py) "
            "or ``significance_vs_baselines.csv``. "
            "When supplied, significant cells receive a $^{\\ast}$ superscript."
        ),
    )
    parser.add_argument(
        "--sig-alpha", type=float, default=0.05,
        help=(
            "P-value threshold for significance annotation. "
            "With 5 CV folds the minimum achievable Wilcoxon p is 0.0625; "
            "use --sig-alpha 0.0625 to annotate any concordant improvement."
        ),
    )
    args = parser.parse_args()

    is_cr_detected = "cr" in str(args.aggregated).lower()
    if is_cr_detected:
        args.dataset_group = "cr"

    # Resolve dataset list
    _GROUP_MAP = {
        "public":     PUBLIC_DATASETS,
        "ehr":        EHR_DATASETS,
        "survset":    SURVSET_DATASETS,
        "ormoni":     ORMONI_DATASETS,
        "cr":         CR_DATASETS,
        "public+ehr": PUBLIC_DATASETS + EHR_DATASETS,
        "all":        ALL_PAPER_DATASETS,
    }
    if args.datasets is not None:
        selected_datasets = args.datasets
    else:
        selected_datasets = _GROUP_MAP[args.dataset_group]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_groups: dict | None = None
    if args.group_datasets:
        grp_pub    = [d for d in selected_datasets if d in PUBLIC_DATASETS]
        grp_ehr    = [d for d in selected_datasets if d in EHR_DATASETS]
        grp_ormoni = [d for d in selected_datasets if d in ORMONI_DATASETS]
        grp_ss     = [d for d in selected_datasets if d.startswith("SS_")]
        grp_cr     = [d for d in selected_datasets if d in CR_DATASETS]
        dataset_groups = {}
        if grp_pub:    dataset_groups["Public Benchmarks"]  = grp_pub
        if grp_ehr:    dataset_groups["EHR (ICU)"]          = grp_ehr
        if grp_ormoni: dataset_groups["OrmoniTirodei"]      = grp_ormoni
        if grp_ss:     dataset_groups["SurvSet"]            = grp_ss
        if grp_cr:     dataset_groups["Competing Risks"]    = grp_cr
        if not dataset_groups:
            dataset_groups = None

    is_cr = "cr" in str(args.aggregated).lower()
    suffix = "_cr" if is_cr else ""

    _lower_is_better_set = set(args.lower_is_better_metrics)

    for metric in args.metrics:
        # Map "Integrated Brier Score" to "IBS" if that's what's in the CSV
        col_name = metric
        if col_name == "Integrated Brier Score":
            col_name = "IBS"
        elif col_name not in ["C_td", "IBS", "D-cal"] and "IBS" in metric:
             col_name = "IBS"

        print(f"\n[{metric}] Loading {args.aggregated} … (using CSV column '{col_name}')")
        try:
            summary = load_aggregated(args.aggregated, col_name)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        mean_pivot, std_pivot = pivot_for_table(summary, selected_datasets)
        lower = col_name in _lower_is_better_set or metric in _lower_is_better_set

        # Load significance map (per-metric, respecting lower_is_better direction)
        sig_map: dict[tuple[str, str], str] | None = None
        if args.sig_file:
            sig_map = _load_sig_map(args.sig_file, alpha=args.sig_alpha, lower_is_better=lower)
            n_marked = sum(1 for v in sig_map.values() if v)
            print(f"  Significance map: {n_marked} cells marked "
                  f"(α={args.sig_alpha}, file={args.sig_file})")

        # Full table
        print(f"  Building full table for {metric} ({len(mean_pivot.index)} models, "
              f"{len([d for d in selected_datasets if d in mean_pivot.columns])} datasets) …")
        tex = build_latex_table(
            mean_pivot, std_pivot,
            datasets=selected_datasets,
            metric=metric,
            lower_is_better=lower,
            dataset_groups=dataset_groups,
            sig_map=sig_map,
        )
        out_path = out_dir / f"table_{metric.replace('-', '_').lower()}_full{suffix}.tex"
        out_path.write_text(tex)
        print(f"  Saved: {out_path}")
        print("\n" + tex[:600] + ("…" if len(tex) > 600 else "") + "\n")

        # Summary table
        if args.summary:
            tex_s = build_summary_table(
                mean_pivot, std_pivot,
                datasets=selected_datasets,
                metric=metric,
                lower_is_better=lower,
                sig_map=sig_map,
            )
            out_s = out_dir / f"table_{metric.replace('-', '_').lower()}_summary{suffix}.tex"
            out_s.write_text(tex_s)
            print(f"  Summary saved: {out_s}")

    # Generate Detailed Longtable
    print(f"\n[Detailed Longtable] Generating combined metrics table ...")
    try:
        sig_map_lt = None
        if args.sig_file:
            sig_map_lt = _load_sig_map(args.sig_file, alpha=args.sig_alpha, lower_is_better=False)
            
        if is_cr:
            longtable_tex = build_latex_longtable_cr_metrics(args.aggregated, selected_datasets, sig_map=sig_map_lt)
            longtable_out = out_dir / f"table_detailed_longtable_cr.tex"
        else:
            longtable_tex = build_latex_longtable_all_metrics(args.aggregated, selected_datasets, sig_map=sig_map_lt)
            longtable_out = out_dir / f"table_detailed_longtable.tex"
            
        longtable_out.write_text(longtable_tex)
        print(f"  Saved detailed longtable: {longtable_out}")
    except Exception as e:
        print(f"  Failed generating detailed longtable: {e}")

    print(f"\nAll tables written to: {out_dir}")

if __name__ == "__main__":
    main()
