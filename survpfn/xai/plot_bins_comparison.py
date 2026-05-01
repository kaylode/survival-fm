"""
xai/plot_bins_comparison.py — Bins-efficiency line plots.

Reads the results directory produced by `benchmark.py --bins` and
generates line plots comparing model performance across different number of bins.

The folder structure is expected to be:

    <results_dir>/<DATASET>/<model>/fold_<N>/bins<B>/metrics.json

Each line in the plot represents a model; the x-axis is the number of bins
and the y-axis is the metric value (mean ± std across folds).

Figures produced
----------------
For each dataset:
    bins_comparison_<DATASET>_<metric>.pdf

Combined (all datasets side-by-side):
    bins_comparison_combined_<metric>.pdf

CLI
---
    uv run python -m survpfn.xai.plot_bins_comparison \
        --results-dir results/benchmark_bins \
        --output-dir xai/figures/benchmark_bins \
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

# ── Styling from analysis.py & plot_label_efficiency.py ──────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 22,
    "axes.titlesize": 26,
    "axes.labelsize": 32,
    "legend.fontsize": 20,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Import styling from analysis if possible, otherwise define fallbacks
try:
    from survpfn.xai.analysis import MODEL_LABELS, DATASET_LABELS, get_style, MODEL_ORDER
except ImportError:
    MODEL_LABELS = {}
    DATASET_LABELS = {}
    MODEL_ORDER = []
    def get_style(m): return {"color": None, "marker": "o", "label": m}

# ── Grouping specific models as in plot_label_efficiency.py ──────────────────
SPECIAL_GROUPS_MAP = {
    "tabpfn_embedding_deephit": "Finetune-DH",
    "tabdpt_embedding_deephit": "Finetune-DH",
    "tabicl_embedding_deephit": "Finetune-DH",
    "tabpfn_finetune": "Finetune-CE",
    "tabdpt_finetune": "Finetune-CE",
    "tabicl_finetune": "Finetune-CE",
    "tabpfn_zeroshot_perbin_time_ens": "Zeroshot",
    "tabdpt_zeroshot_perbin_time_ens": "Zeroshot",
    "tabicl_zeroshot_perbin_time_ens": "Zeroshot",
}

HIGHLIGHTED_GROUPS = ["Finetune-DH", "Finetune-CE", "Zeroshot"]

MODEL_COLORS = {
    "Finetune-DH": "#71b7e7", # Blue-ish
    "Finetune-CE": "#ed8076", # Red-ish
    "Zeroshot": "#2CA02C",    # Green
}

MODEL_LABELS.update({
    "Finetune-DH": "Finetune-DeepHit",
    "Finetune-CE": "Finetune-CE",
    "Zeroshot": "Zero-shot",
})

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_bins_results(results_dir: str, benchmark_dir: str = "results/benchmark") -> pd.DataFrame:
    """Collect metrics from a bins-based benchmark run."""
    root = Path(results_dir)
    records: list[dict] = []

    # Path pattern: <results_dir>/<DATASET>/<model>/fold_<N>/bins<B>/metrics.json
    for metrics_path in sorted(root.glob("*/*/fold_*/bins*/metrics.json")):
        parts = metrics_path.parts
        dataset = parts[-5]
        model = parts[-4]
        fold_str = parts[-3]
        bin_str = parts[-2]

        try:
            fold = int(fold_str.replace("fold_", ""))
        except (ValueError, IndexError):
            fold = -1
            
        try:
            bins = int(bin_str.replace("bins", ""))
        except (ValueError, IndexError):
            continue

        try:
            with metrics_path.open() as f:
                metrics = json.load(f)
        except Exception:
            continue

        row: dict = {
            "Dataset":  dataset,
            "Model":    model,
            "Bins":     bins,
            "Fold":     fold,
        }
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
                row[k] = np.nan
            else:
                row[k] = v

        records.append(row)

    # ── Collect 100-bin results from benchmark_dir (treated as bins=100) ──────
    bins_pairs = {(r["Dataset"], r["Model"]) for r in records}
    bench_root = Path(benchmark_dir)
    if bench_root.exists():
        for ds, model in bins_pairs:
            model_dir = bench_root / ds / model
            if not model_dir.exists():
                continue
            
            for metrics_path in model_dir.glob("fold_*/metrics.json"):
                # Avoid collecting from 'benchmark_bins' if it happens to be a child
                if "benchmark_bins" in metrics_path.parts: continue

                parts = metrics_path.parts
                fold_str = parts[-2]
                try:
                    fold = int(fold_str.split("_")[1])
                except (IndexError, ValueError):
                    fold = -1
                
                try:
                    with metrics_path.open() as f:
                        metrics = json.load(f)
                except Exception:
                    continue

                row: dict = {
                    "Dataset":  ds,
                    "Model":    model,
                    "Bins":     100, # Treat benchmark results as 100 bins
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
            f"No bins-based metrics.json found under '{results_dir}'. "
            "Ensure the directory structure is <DATASET>/<model>/fold_<N>/bins<B>/metrics.json"
        )

    df = pd.DataFrame(records)
    
    # Filter only models in MODEL_ORDER
    if MODEL_ORDER:
        df = df[df["Model"].isin(MODEL_ORDER)]
        
    df = df.sort_values(["Dataset", "Model", "Bins", "Fold"]).reset_index(drop=True)
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

def plot_bins_line(
    df: pd.DataFrame,
    metric: str,
    dataset: str,
    output_dir: Path,
    lower_is_better: bool = False,
    figsize: tuple[float, float] = (12, 10),
) -> None:
    """Single-dataset line plot: metric vs bins, one line per model."""
    sub = df[(df["Dataset"] == dataset) & df[metric].notna()].copy()
    if sub.empty:
        print(f"  Skipping {dataset}/{metric}: no data.")
        return

    # Aggregate across folds
    agg = (
        sub.groupby(["Model", "Bins"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=figsize)
    bins_list = sorted(agg["Bins"].unique())
    bins_to_idx = {b: i for i, b in enumerate(bins_list)}

    models = sorted(agg["Model"].unique())
    markers = ["o", "s", "D", "^", "v", "p", "h", "8", "*", "X", "P", "d"]

    for i, model in enumerate(models):
        m = agg[agg["Model"] == model].sort_values("Bins")
        
        display = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model)
        marker = markers[i % len(markers)]
        
        # ── Styling from plot_label_efficiency.py ────────────────────────────
        is_highlight = model in HIGHLIGHTED_GROUPS
        alpha = 1.0 if is_highlight else 0.5
        lw    = 4.0 if is_highlight else 2.5
        z     = 10  if is_highlight else 2
        ms    = 12  if is_highlight else 6
        # ─────────────────────────────────────────────────────────────────────

        line_args = {
            "marker": marker,
            "markersize": ms,
            "linewidth": lw,
            "label": display,
            "alpha": alpha,
            "zorder": z
        }
        if color:
            line_args["color"] = color

        ax.plot(
            m["Bins"].map(bins_to_idx), m["mean"],
            **line_args
        )
        
        fill_color = color if color else ax.get_lines()[-1].get_color()
        ax.fill_between(
            m["Bins"].map(bins_to_idx),
            m["mean"] - m["std"],
            m["mean"] + m["std"],
            alpha=alpha * 0.15, color=fill_color, zorder=z-1
        )

    metric_display = "C-index" if metric == 'C_td' else metric
    ax.set_xlabel("Number of Bins", fontsize=28)
    ax.set_ylabel(metric_display, fontsize=28)
    
    ax.set_xticks(range(len(bins_list)))
    ax.set_xticklabels([str(b) for b in bins_list])
    ax.tick_params(axis='both', which='major', labelsize=24)

    ax.legend(
        loc="best",
        fontsize=18,
        frameon=True, framealpha=0.9, edgecolor="0.8",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    sns.despine(ax=ax)
    
    title = f"{DATASET_LABELS.get(dataset, dataset)}"
    ax.set_title(title, fontweight="bold", pad=20)
    
    fig.tight_layout()

    fname = f"bins_comparison_{dataset}_{metric.lower().replace('-', '_')}.pdf"
    _save(fig, output_dir / fname)

def plot_bins_combined(
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

    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7), sharey=False, squeeze=False)
    axes = axes.flatten()

    all_models = sorted(df["Model"].unique())
    all_bins = sorted(df["Bins"].unique())
    bins_to_idx = {b: i for i, b in enumerate(all_bins)}
    
    handles, labels = [], []
    markers = ["o", "s", "D", "^", "v", "p", "h", "8", "*", "X", "P", "d"]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        sub = df[(df["Dataset"] == dataset) & df[metric].notna()].copy()
        if sub.empty:
            ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)} (no data)")
            continue

        agg = (
            sub.groupby(["Model", "Bins"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        for i, model in enumerate(all_models):
            m = agg[agg["Model"] == model].sort_values("Bins")
            if m.empty:
                continue
            
            display = MODEL_LABELS.get(model, model)
            color = MODEL_COLORS.get(model)
            marker = markers[i % len(markers)]

            is_highlight = model in HIGHLIGHTED_GROUPS
            alpha = 1.0 if is_highlight else 0.25
            lw    = 5.0 if is_highlight else 1.5
            z     = 10  if is_highlight else 2
            ms    = 12  if is_highlight else 6

            line_args = {
                "marker": marker,
                "markersize": ms,
                "linewidth": lw,
                "label": display,
                "alpha": alpha,
                "zorder": z
            }
            if color:
                line_args["color"] = color

            line, = ax.plot(
                m["Bins"].map(bins_to_idx), m["mean"],
                **line_args
            )
            
            fill_color = color if color else line.get_color()
            ax.fill_between(
                m["Bins"].map(bins_to_idx),
                m["mean"] - m["std"],
                m["mean"] + m["std"],
                alpha=alpha * 0.15, color=fill_color, zorder=z-1
            )

            if idx == 0:
                handles.append(line)
                labels.append(display)

        ax.set_xlabel("Number of Bins", fontsize=24)
        if idx == 0:
            metric_display = "C-index" if metric == 'C_td' else metric
            ax.set_ylabel(metric_display, fontsize=24)
        
        ax.set_xticks(range(len(all_bins)))
        ax.set_xticklabels([str(b) for b in all_bins])
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        sns.despine(ax=ax)

    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(5, len(handles)),
        frameon=True, framealpha=0.9, edgecolor="0.8",
        bbox_to_anchor=(0.5, -0.05),
    )
    
    fig.suptitle(f"Bins Comparison — {metric}", fontsize=28, fontweight="bold", y=1.02)
    fig.tight_layout()

    fname = f"bins_comparison_combined_{metric.lower().replace('-', '_')}.pdf"
    _save(fig, output_dir / fname)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot bins-comparison line charts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", default="/home/mpham/workspace/source/ehrfm/survpfn/results/benchmark_bins",
        help="Root results directory.",
    )
    parser.add_argument(
        "--benchmark-dir", default="/home/mpham/workspace/source/ehrfm/survpfn/results/benchmark",
        help="Main benchmark directory for bins=100 results.",
    )
    parser.add_argument(
        "--output-dir", default="/home/mpham/workspace/source/ehrfm/survpfn/results/xai/figures/benchmark_bins",
        help="Directory where PDF figures are saved.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["C_td", "IBS"],
        help="Metrics to plot.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of datasets to plot.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting results from '{args.results_dir}' (and '{args.benchmark_dir}' for 100 bins)...")
    try:
        df = collect_bins_results(args.results_dir, args.benchmark_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # ── Grouping specific models as requested ────────────────────────────────
    df["ModelGroup"] = df["Model"].map(SPECIAL_GROUPS_MAP).fillna(df["Model"])
    
    # Aggregate across the models within each group
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ["Bins", "Fold"]:
        if col in numeric_cols: numeric_cols.remove(col)
    
    df = df.groupby(["Dataset", "ModelGroup", "Bins", "Fold"])[numeric_cols].mean().reset_index()
    df = df.rename(columns={"ModelGroup": "Model"})

    print(f"  {len(df)} rows | {df['Dataset'].nunique()} datasets "
          f"| {df['Model'].nunique()} models "
          f"| bins: {sorted(df['Bins'].unique())}")

    if args.datasets:
        df = df[df["Dataset"].isin(args.datasets)]

    lower_is_better = {"IBS", "D-cal", "D-cal_pval", "MAE-Margin", "MAE-PO", "BS_q25", "BS_q50", "BS_q75"}

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"  Skipping {metric}: column not found.")
            continue
            
        print(f"Plotting {metric}...")
        
        # Per-dataset figures
        for dataset in sorted(df["Dataset"].unique()):
            plot_bins_line(
                df, metric, dataset, output_dir,
                lower_is_better=(metric in lower_is_better),
            )

        # Combined figure
        if df["Dataset"].nunique() > 1:
            plot_bins_combined(
                df, metric, output_dir,
                lower_is_better=(metric in lower_is_better),
            )

    print(f"All figures saved to '{output_dir}'.")

if __name__ == "__main__":
    main()
