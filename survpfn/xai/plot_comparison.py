"""
xai/plot_comparison.py — Generate comparison figures from aggregated benchmark results.

Reads the CSV produced by scripts/aggregate.py (or scripts/benchmark.py) and
generates PDF figures in the specified output directory.

Figures produced
----------------
model_comparison_cindex.pdf  — bar chart: C-index per model, grouped by dataset
model_comparison_ibs.pdf     — bar chart: IBS per model (lower is better)
model_comparison_auc.pdf     — bar chart: mean time-dependent AUC
statistical_tests.csv/pdf    — Wilcoxon signed-rank results vs Cox baseline
feature_importance_<dataset>_<model>.pdf
                             — horizontal bar chart per (dataset, model) that
                               has feature_importance.json files

CLI
---
    uv run python xai/plot_comparison.py --aggregated results/aggregated.csv
    uv run python xai/plot_comparison.py \\
        --aggregated results/aggregated.csv \\
        --results-dir results \\
        --output-dir xai/figures \\
        --datasets GBSG METABRIC \\
        --top-k 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path, close: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")
    if close:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Metric bar charts
# ---------------------------------------------------------------------------

def plot_metric(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    datasets: list[str] | None = None,
    figsize: tuple[int, int] = (14, 6),
    lower_is_better: bool = False,
) -> None:
    """Grouped bar chart of *metric* across models and datasets."""
    if metric not in df.columns:
        print(f"  Skipping {metric}: column not found in aggregated CSV.")
        return

    sub = df.dropna(subset=[metric])
    if datasets:
        sub = sub[sub["Dataset"].isin(datasets)]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=sub, x="Dataset", y=metric, hue="Model",
        estimator="mean", errorbar="sd", capsize=0.05, ax=ax,
    )
    direction = "(↓ better)" if lower_is_better else "(↑ better)"
    ax.set_title(f"Model comparison — {metric} {direction}")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()

    fname = f"model_comparison_{metric.lower().replace('-', '_')}.pdf"
    _save(fig, output_dir / fname)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_statistical_tests(
    df: pd.DataFrame,
    metric: str = "C-index",
    reference: str = "cox",
    output_dir: Path = Path("xai/figures"),
) -> pd.DataFrame:
    """Wilcoxon signed-rank test: each model vs *reference* within each dataset."""
    records = []
    for dataset in df["Dataset"].unique():
        ds = df[df["Dataset"] == dataset]
        ref = ds[ds["Model"] == reference].sort_values("Fold")[metric].values
        if len(ref) == 0:
            continue

        for model in ds["Model"].unique():
            if model == reference:
                continue
            mod = ds[ds["Model"] == model].sort_values("Fold")[metric].values
            if len(mod) != len(ref) or len(ref) < 2:
                continue
            diff = mod - ref
            if np.all(diff == 0):
                p = 1.0
            else:
                try:
                    _, p = wilcoxon(mod, ref, zero_method="zsplit")
                except Exception:
                    p = float("nan")
            records.append({
                "Dataset":    dataset,
                "Model":      model,
                "Reference":  reference,
                "Mean_diff":  float(np.mean(diff)),
                "p-value":    p,
                "Significant": bool(p < 0.05) if not np.isnan(p) else False,
            })

    results = pd.DataFrame(records)
    if results.empty:
        return results

    csv_path = output_dir / "statistical_tests.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Heatmap: mean diff (vs reference)
    pivot = results.pivot_table(index="Model", columns="Dataset", values="Mean_diff")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.6)))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", center=0,
            cmap="RdYlGn", linewidths=0.5, ax=ax,
        )
        ax.set_title(f"Mean Δ{metric} vs {reference} (green = better)")
        fig.tight_layout()
        _save(fig, output_dir / "statistical_tests_heatmap.pdf")

    return results


# ---------------------------------------------------------------------------
# Feature importance plots
# ---------------------------------------------------------------------------

def plot_feature_importance(
    results_dir: Path,
    output_dir: Path,
    datasets: list[str] | None = None,
    top_k: int = 20,
) -> None:
    """Plot feature importance for each (dataset, model) that has the JSON file."""
    for fi_path in sorted(results_dir.glob("*/*/fold_1/feature_importance.json")):
        parts = fi_path.parts
        dataset = parts[-4]
        model   = parts[-3]

        if datasets and dataset not in datasets:
            continue

        with fi_path.open() as f:
            fi = json.load(f)

        fi_type = fi.get("type", "unknown")

        if fi_type == "impurity":
            feat_dict = fi.get("features", {})
            if not feat_dict:
                continue
            items = sorted(feat_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            names, vals = zip(*items)

            fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
            ax.barh(names[::-1], vals[::-1], color="steelblue")
            ax.set_xlabel("Feature importance")
            ax.set_title(f"{dataset} — {model} (fold 1, impurity)")
            fig.tight_layout()
            _save(fig, output_dir / f"feature_importance_{dataset}_{model}.pdf")

        elif fi_type == "cox_coefficients":
            coef = fi.get("exp_coef", fi.get("coef", {}))
            p    = fi.get("p", {})
            if not coef:
                continue
            # Keep only significant features (p < 0.05) or top_k
            df_fi = pd.DataFrame({"feature": list(coef.keys()),
                                  "exp_coef": list(coef.values()),
                                  "p": [p.get(k, 1.0) for k in coef]})
            df_fi = df_fi[df_fi["p"] < 0.05].sort_values("exp_coef")
            if df_fi.empty:
                df_fi = pd.DataFrame({"feature": list(coef.keys()),
                                      "exp_coef": list(coef.values())}) \
                          .sort_values("exp_coef").tail(top_k)
            df_fi = df_fi.tail(top_k)

            fig, ax = plt.subplots(figsize=(8, max(4, len(df_fi) * 0.35)))
            colors = ["tomato" if v > 1 else "steelblue" for v in df_fi["exp_coef"]]
            ax.barh(df_fi["feature"], df_fi["exp_coef"], color=colors)
            ax.axvline(x=1, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Hazard Ratio (exp(coef))")
            ax.set_title(f"{dataset} — {model} Cox (fold 1, p < 0.05)")
            fig.tight_layout()
            _save(fig, output_dir / f"feature_importance_{dataset}_{model}.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison figures from aggregated benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--aggregated", default="results/aggregated.csv",
        help="Path to the aggregated CSV from scripts/aggregate.py.",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Root results directory (for locating feature_importance.json files).",
    )
    parser.add_argument(
        "--output-dir", default="xai/figures",
        help="Directory where PDF figures are saved.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of datasets to plot. Defaults to all.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["C-index", "IBS", "AUC_mean"],
        help="Metrics to plot.",
    )
    parser.add_argument(
        "--reference", default="cox",
        help="Reference model for Wilcoxon statistical tests.",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Max features to show in importance plots.",
    )
    parser.add_argument(
        "--no-feature-importance", action="store_true",
        help="Skip feature importance plots.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading aggregated results from '{args.aggregated}' ...")
    df = pd.read_csv(args.aggregated)
    print(f"  {len(df)} rows | {df['Dataset'].nunique()} datasets "
          f"| {df['Model'].nunique()} models")

    lower_is_better = {"IBS", "D-cal"}
    for metric in args.metrics:
        print(f"\nPlotting {metric} ...")
        plot_metric(
            df, metric, output_dir,
            datasets=args.datasets,
            lower_is_better=(metric in lower_is_better),
        )

    print("\nRunning statistical tests ...")
    run_statistical_tests(df, metric="C-index", reference=args.reference,
                          output_dir=output_dir)

    if not args.no_feature_importance:
        print("\nPlotting feature importances ...")
        plot_feature_importance(
            results_dir=Path(args.results_dir),
            output_dir=output_dir,
            datasets=args.datasets,
            top_k=args.top_k,
        )

    print(f"\nAll figures saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
