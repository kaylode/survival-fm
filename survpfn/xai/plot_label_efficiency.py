"""
xai/plot_label_efficiency.py — Label-efficiency line plots.

Reads the results directory produced by `benchmark.py --label-fractions` and
generates line plots comparing model performance across training-data fractions.

The folder structure is expected to be:

    <results_dir>/<DATASET>/<model>_frac<F>/fold_<N>/metrics.json

Each line in the plot represents a model; the x-axis is the label fraction
and the y-axis is the metric value (mean ± std across folds).

Figures produced
----------------
For each dataset:
    label_efficiency_<DATASET>_<metric>.pdf

Combined (all datasets side-by-side):
    label_efficiency_combined_<metric>.pdf

CLI
---
    uv run python -m survpfn.xai.plot_label_efficiency \
        --results-dir results/benchmark_frac \
        --output-dir xai/figures/label_efficiency \
        --metrics C_td IBS AUC_mean
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Plotting defaults ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Model display names & styling ────────────────────────────────────────────
MODEL_ORDER = [
    "cox", "rsf", "gbsa",
    "deepsurv", "mtlr",  "deephit_single", "soden", "dysurv", "survtrace", 

    "tabpfn_zeroshot_perbin_time_ens", "tabdpt_zeroshot_perbin_time_ens", "tabicl_zeroshot_perbin_time_ens",
    "tabpfn_finetune", "tabdpt_finetune", "tabicl_finetune",

    "tabpfn_embedding_mtlr", "tabdpt_embedding_mtlr", "tabicl_embedding_mtlr",
    "tabpfn_embedding_deephit", "tabdpt_embedding_deephit", "tabicl_embedding_deephit",
    "tabpfn_embedding_cox", "tabdpt_embedding_cox", "tabicl_embedding_cox",

    "cox_cr",  "fine_gray_cr",
    "deephit_cr", 
    "tabpfn_zeroshot_cr_multinomial", "tabpfn_zeroshot_cr_perevent",
    "tabpfn_embedding_deephit_cr", "tabdpt_embedding_deephit_cr", "tabicl_embedding_deephit_cr",
    "tabpfn_embedding_deephit_v2_cr", "tabdpt_embedding_deephit_v2_cr", "tabicl_embedding_deephit_v2_cr",
    "tabpfn_embedding_deephit_v2_cr_adapter", "tabdpt_embedding_deephit_v2_cr_adapter", "tabicl_embedding_deephit_v2_cr_adapter",
    "tabpfn_embedding_cox_cr", "tabdpt_embedding_cox_cr", "tabicl_embedding_cox_cr",
    "tabpfn_embedding_cox_cr_adapter", "tabdpt_embedding_cox_cr_adapter", "tabicl_embedding_cox_cr_adapter",
]

MODEL_LABELS = {
    "km": "KM", "cox": "Cox PH", "rsf": "RSF", "gbsa": "GBSA",
    "deepsurv": "DeepSurv", "mtlr": "MLP-MTLR", "deephit_single": "MLP-DeepHit",
    "survtrace": "SurvTrace", "soden": "SODEN", "dysurv": "DySurv", "beta_surv": "Beta-Surv",
    "tabpfn_zeroshot_perbin_time_ens": "TabPFN-ZS", "tabdpt_zeroshot_perbin_time_ens": "TabDPT-ZS", "tabicl_zeroshot_perbin_time_ens": "TabICL-ZS",
    "tabpfn_finetune": "TabPFN-FT-CE", "tabdpt_finetune": "TabDPT-FT-CE", "tabicl_finetune": "TabICL-FT-CE",
    "cox_cr": "Cox-CR", "aj_cr": "Aalen-Johansen", "fine_gray_cr": "Fine-Gray",
    "survival_boost_cr": "SurvBoost-CR", "deephit_cr": "DeepHit-CR",
    "tabpfn_zeroshot_cr_multinomial": "TabPFN-ZS-CR-Mul", "tabpfn_zeroshot_cr_perevent": "TabPFN-ZS-CR-PE",
}

for fm, fm_name in zip(["tabpfn", "tabdpt", "tabicl"], ["TabPFN", "TabDPT", "TabICL"]):
    for task, task_name in zip(["embedding", "joint"], ["FT", "Joint"]):
        for head, h_name in zip(["cox", "deephit", "mtlr"], ["Cox", "DeepHit", "MTLR"]):
            MODEL_LABELS[f"{fm}_{task}_{head}"] = f"{fm_name}-{task_name}-{h_name}"
            MODEL_LABELS[f"{fm}_{task}_{head}_adapter"] = f"{fm_name}-{task_name}-{h_name}+Adp"
        for head, h_name in zip(["deephit_cr", "deephit_v2_cr", "cox_cr"], ["DH-CR", "DH-v2-CR", "Cox-CR"]):
            MODEL_LABELS[f"{fm}_{task}_{head}"] = f"{fm_name}-{task_name}-{h_name}"
            MODEL_LABELS[f"{fm}_{task}_{head}_adapter"] = f"{fm_name}-{task_name}-{h_name}+Adp"

MODEL_GROUPS = {
    "Baseline": ["km", "cox", "cox_cr", "aj_cr", "fine_gray_cr"], "Tree": ["rsf", "gbsa"],
    "Deep": ["deepsurv", "mtlr", "deephit_single", "dysurv", "deephit_cr"], "Attention": ["survtrace"],
    "Finetune-Cox": [m for m in MODEL_ORDER if "cox" in m and ("_embedding" in m or "_joint" in m)],
    "Finetune-DeepHit": [m for m in MODEL_ORDER if "deephit" in m and ("_embedding" in m or "_joint" in m)],
    "Finetune-MTLR": [m for m in MODEL_ORDER if "mtlr" in m and ("_embedding" in m or "_joint" in m)],
    "Zero-shot": ["tabpfn_zeroshot_perbin_time_ens", "tabdpt_zeroshot_perbin_time_ens", "tabicl_zeroshot_perbin_time_ens", "tabpfn_zeroshot_cr_multinomial", "tabpfn_zeroshot_cr_perevent"],
    "Finetune-CE": ["tabpfn_finetune", "tabdpt_finetune", "tabicl_finetune"],
}
MODEL_TO_GROUP = {m: g for g, ms in MODEL_GROUPS.items() for m in ms}

DATASET_LABELS = {
    "SUPPORT2": "SUPPORT2", "METABRIC": "METABRIC", "GBSG": "GBSG", "FLCHAIN": "FL-Chain",
    "VETERANS": "Veterans", "WHAS500": "WHAS500", "SEER": "SEER",
    "ORMONI_TIRODEI": "OrmoniTirodei", "ORMONI_TIRODEI_MORTALITY": "Orm.Tir.-Mort.",
    "ORMONI_TIRODEI_CV": "Orm.Tir.-CV", "ORMONI_TIRODEI_MI": "Orm.Tir.-MI",
    "ORMONI_TIRODEI_STROKE": "Orm.Tir.-Stroke", "FRAMINGHAM": "Fram-CR",
    "PBC2": "PBC2-CR", "SUPPORT_CR": "Supp-CR", "SYNTHETIC_CR": "Syn-CR",
    "EICU_SURV": "eICU", "MIMIC_SURV_B": "MIMIC-IV",
}

# Build a consistent color map per model for line plots (ordered by group)
_ALL_MODELS_ORDERED = []
for _cat_models in MODEL_GROUPS.values():
    _ALL_MODELS_ORDERED.extend([m for m in _cat_models if m in MODEL_ORDER])

_COLORS = sns.color_palette("husl", n_colors=max(len(_ALL_MODELS_ORDERED), 15))
MODEL_COLORS: dict[str, tuple] = {
    m: _COLORS[i % len(_COLORS)] for i, m in enumerate(_ALL_MODELS_ORDERED)
}

# Marker cycle
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "p", "h", "<", ">"]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

_FRAC_PATTERN = re.compile(r"^(.+)_frac(\d+\.\d+)$")


def collect_fraction_results(results_dir: str) -> pd.DataFrame:
    """Collect metrics from a fraction-based benchmark run.

    Parses model tags of the form ``<model>_frac<F>`` and returns a DataFrame
    with columns: Dataset, Model, Fraction, Fold, + metric columns.
    """
    root = Path(results_dir)
    records: list[dict] = []

    for metrics_path in sorted(root.glob("*/*/fold_*/metrics.json")):
        parts = metrics_path.parts
        dataset   = parts[-4]
        model_tag = parts[-3]
        fold_str  = parts[-2]

        try:
            fold = int(fold_str.split("_")[1])
        except (IndexError, ValueError):
            fold = -1

        # Parse model_tag → (model_name, fraction)
        match = _FRAC_PATTERN.match(model_tag)
        if match is None:
            # Not a fraction experiment entry — skip
            continue
        model_name = match.group(1)
        fraction   = float(match.group(2))

        with metrics_path.open() as f:
            metrics = json.load(f)

        row: dict = {
            "Dataset":  dataset,
            "Model":    model_name,
            "Fraction": fraction,
            "Fold":     fold,
        }
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
                row[k] = np.nan
            else:
                row[k] = v

        records.append(row)

    if not records:
        raise FileNotFoundError(
            f"No fraction-labelled metrics.json found under '{results_dir}'. "
            "Ensure benchmark.py was run with --label-fractions."
        )

    df = pd.DataFrame(records)
    df = df.sort_values(["Dataset", "Model", "Fraction", "Fold"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path, close: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")
    if close:
        plt.close(fig)


def plot_fraction_line(
    df: pd.DataFrame,
    metric: str,
    dataset: str,
    output_dir: Path,
    lower_is_better: bool = False,
    figsize: tuple[float, float] = (8, 5.5),
) -> None:
    """Single-dataset line plot: metric vs fraction, one line per model."""
    sub = df[(df["Dataset"] == dataset) & df[metric].notna()].copy()
    if sub.empty:
        print(f"  Skipping {dataset}/{metric}: no data.")
        return

    # Aggregate across folds
    agg = (
        sub.groupby(["Model", "Fraction"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)

    models = sorted(agg["Model"].unique())
    for i, model in enumerate(models):
        m = agg[agg["Model"] == model].sort_values("Fraction")
        display = MODEL_LABELS.get(model, model)
        color   = MODEL_COLORS.get(model, _COLORS[i % len(_COLORS)])
        marker  = _MARKERS[i % len(_MARKERS)]

        ax.plot(
            m["Fraction"], m["mean"],
            marker=marker, markersize=6, linewidth=2,
            label=display, color=color,
        )
        ax.fill_between(
            m["Fraction"],
            m["mean"] - m["std"],
            m["mean"] + m["std"],
            alpha=0.12, color=color,
        )

    ax.set_xlabel("Training Label Fraction")
    ax.set_ylabel(metric)
    direction = " (↓ better)" if lower_is_better else " (↑ better)"
    ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — {metric}{direction}")

    # Use percentage tick labels
    ax.set_xticks(sorted(agg["Fraction"].unique()))
    ax.set_xticklabels([f"{f*100:g}%" for f in sorted(agg["Fraction"].unique())])

    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        frameon=True, framealpha=0.9, edgecolor="0.8",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    sns.despine(ax=ax)
    fig.tight_layout()

    fname = f"label_efficiency_{dataset}_{metric.lower().replace('-', '_')}.pdf"
    _save(fig, output_dir / fname)


def plot_fraction_combined(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    lower_is_better: bool = False,
) -> None:
    """Side-by-side subplots: one per dataset, all sharing the same legend."""
    datasets = sorted(df["Dataset"].unique())
    n = len(datasets)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5), sharey=True, squeeze=False)
    axes = axes.flatten()

    all_models = [m for m in MODEL_ORDER if m in df["Model"].unique()]
    all_models.extend(sorted([m for m in df["Model"].unique() if m not in all_models]))
    
    handles, labels = [], []

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        sub = df[(df["Dataset"] == dataset) & df[metric].notna()].copy()
        if sub.empty:
            ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} (no data)")
            continue

        agg = (
            sub.groupby(["Model", "Fraction"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        for i, model in enumerate(all_models):
            m = agg[agg["Model"] == model].sort_values("Fraction")
            if m.empty:
                continue
            display = MODEL_LABELS.get(model, model)
            color   = MODEL_COLORS.get(model, _COLORS[i % len(_COLORS)])
            marker  = _MARKERS[i % len(_MARKERS)]

            line, = ax.plot(
                m["Fraction"], m["mean"],
                marker=marker, markersize=6, linewidth=2,
                label=display, color=color,
            )
            ax.fill_between(
                m["Fraction"],
                m["mean"] - m["std"],
                m["mean"] + m["std"],
                alpha=0.12, color=color,
            )

            # Collect legend handles from first subplot only
            if idx == 0:
                handles.append(line)
                labels.append(display)

        ax.set_xlabel("Training Label Fraction")
        if idx == 0:
            ax.set_ylabel(metric)
        ax.set_xticks(sorted(agg["Fraction"].unique()))
        ax.set_xticklabels([f"{f*100:g}%" for f in sorted(agg["Fraction"].unique())])
        direction = " (↓)" if lower_is_better else " (↑)"
        ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}{direction}")
        ax.grid(True, alpha=0.3, linestyle="--")
        sns.despine(ax=ax)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        frameon=True, framealpha=0.9, edgecolor="0.8",
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(f"Label Efficiency — {metric}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    fname = f"label_efficiency_combined_{metric.lower().replace('-', '_')}.pdf"
    _save(fig, output_dir / fname)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot label-efficiency line charts from fraction-based benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", default="results/benchmark_frac",
        help="Root results directory produced by benchmark.py --label-fractions.",
    )
    parser.add_argument(
        "--output-dir", default="results/xai/label_efficiency",
        help="Directory where PDF figures are saved.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["C_td", "IBS", "AUC_mean"],
        help="Metrics to plot.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of datasets to plot. Defaults to all found.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting fraction results from '{args.results_dir}' ...")
    df = collect_fraction_results(args.results_dir)
    print(f"  {len(df)} rows | {df['Dataset'].nunique()} datasets "
          f"| {df['Model'].nunique()} models "
          f"| fractions: {sorted(df['Fraction'].unique())}")

    if args.datasets:
        df = df[df["Dataset"].isin(args.datasets)]

    lower_is_better = {"IBS", "D-cal", "D-cal_pval", "MAE-Margin", "MAE-PO"}

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"\n  Skipping {metric}: column not found.")
            continue
        print(f"\nPlotting {metric} ...")

        # Per-dataset figures
        for dataset in sorted(df["Dataset"].unique()):
            plot_fraction_line(
                df, metric, dataset, output_dir,
                lower_is_better=(metric in lower_is_better),
            )

        # Combined figure
        plot_fraction_combined(
            df, metric, output_dir,
            lower_is_better=(metric in lower_is_better),
        )

    # Also save the aggregated fraction data as CSV for reference
    csv_path = output_dir / "fraction_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Aggregated fraction data saved to: {csv_path}")

    print(f"\nAll figures saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
