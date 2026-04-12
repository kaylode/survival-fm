"""
scripts/aggregate.py — Aggregate per-run result folders into a single CSV.

Walks the output directory produced by scripts/benchmark.py and collects
every ``metrics.json`` file into a flat DataFrame, tagged with dataset,
model, and fold derived from the folder path structure:

    <output_dir>/<DATASET>/<model>/fold_<N>/metrics.json

The aggregated CSV is written to ``<output_dir>/aggregated.csv``.
A summary table (mean ± std per dataset/model for C-index) is printed
to stdout and saved as ``<output_dir>/summary.csv``.

CLI
---
    uv run python -m survpfn.scripts.aggregate
    uv run python -m survpfn.scripts.aggregate --output-dir results --dest aggregated.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect_results(output_dir: str = "results") -> pd.DataFrame:
    """Glob all metrics.json files and build a flat DataFrame.

    Expected folder layout
    ----------------------
    ``<output_dir>/<DATASET>/<model>/fold_<N>/metrics.json``

    Additional files parsed if present (merged into the row):
    - ``metadata.json`` — run config (timestamp, n_train, n_test, n_features, ...)

    Returns
    -------
    pd.DataFrame with columns: Dataset, Model, Fold, + all metric keys.
    """
    root = Path(output_dir)
    records = []

    for metrics_path in sorted(root.glob("*/*/fold_*/metrics.json")):
        parts = metrics_path.parts
        # Layout: root / DATASET / model / fold_N / metrics.json
        dataset = parts[-4]
        model   = parts[-3]
        fold_str = parts[-2]            # "fold_1", "fold_2", ...

        try:
            fold = int(fold_str.split("_")[1])
        except (IndexError, ValueError):
            fold = -1

        with metrics_path.open() as f:
            metrics = json.load(f)

        row = {"Dataset": dataset, "Model": model, "Fold": fold}

        # Merge metadata if present
        meta_path = metrics_path.parent / "metadata.json"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
            for key in ("n_train", "n_test", "n_features", "tune", "timestamp"):
                if key in meta:
                    row[key] = meta[key]

        # Convert null → NaN for numeric columns
        for k, v in metrics.items():
            row[k] = float("nan") if v is None else v

        records.append(row)

    if not records:
        raise FileNotFoundError(
            f"No metrics.json files found under '{output_dir}'. "
            "Run scripts/benchmark.py first."
        )

    df = pd.DataFrame(records)

    # Consistent ordering
    id_cols = ["Dataset", "Model", "Fold"]
    other_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + other_cols].sort_values(["Dataset", "Model", "Fold"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame, metric: str = "C-index") -> pd.DataFrame:
    """Return mean ± std per (Dataset, Model) for *metric*."""
    if metric not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(["Dataset", "Model"])[metric]
        .agg(["mean", "std", "count"])
        .round(4)
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n_folds"})
        .sort_values(["Dataset", f"{metric}_mean"], ascending=[True, False])
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate SurvPFN per-run result folders into a CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default="results/benchmark",
        help="Root directory produced by scripts/benchmark.py.",
    )
    parser.add_argument(
        "--dest", default=None,
        help="Output CSV path. Defaults to <output-dir>/aggregated.csv.",
    )
    parser.add_argument(
        "--summary-metric", default="C_td",
        help="Metric to include in the printed summary table.",
    )
    args = parser.parse_args()

    print(f"Collecting results from '{args.output_dir}' ...")
    df = collect_results(args.output_dir)
    print(f"  Found {len(df)} fold-level records across "
          f"{df['Dataset'].nunique()} datasets and {df['Model'].nunique()} models.")

    dest = args.dest or str(Path(args.output_dir) / "aggregated.csv")
    df.to_csv(dest, index=False)
    print(f"  Aggregated results saved to: {dest}")

    summary = summarize(df, metric=args.summary_metric)
    if not summary.empty:
        summary_path = str(Path(args.output_dir) / "summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary ({args.summary_metric} mean ± std):\n")
        print(summary.to_string(index=False))
        print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
