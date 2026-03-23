"""
src/evaluate.py — End-to-end evaluation pipeline.

This module ties together data preparation, model training, metric computation,
statistical testing, and result visualisation into a single callable pipeline.

Typical usage
-------------
    from src.evaluate import run_evaluation_pipeline

    run_evaluation_pipeline(
        data_dir="Dataset Sirbu",
        results_dir="results",
        n_splits=5,
    )

Pipeline steps
--------------
1. Load and merge raw data  (``src.data.load_and_merge_data``)
2. Clean and impute          (``src.data.clean_and_impute``)
3. For each task:
   a. Task-specific feature preparation  (``src.data.prepare_*``)
   b. K-fold cross-validation            (``src.train.cross_validate``)
   c. Metric aggregation                 (``src.metrics.summarise_results``)
4. Statistical comparison vs. CoxPH      (``src.metrics.run_statistical_tests``)
5. Plots saved as PDFs in ``results_dir``
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data import (
    clean_and_impute,
    load_and_merge_data,
    prepare_cardiovascular_data,
    prepare_cox_data,
    prepare_mi_data,
    prepare_stroke_data,
)
from src.metrics import run_statistical_tests, summarise_results
from src.models import CoxPHWrapper, DeepHitWrapper, DeepSurvWrapper, GBSAWrapper, RSFWrapper
from src.train import cross_validate


# ---------------------------------------------------------------------------
# Default task registry
# ---------------------------------------------------------------------------

DEFAULT_TASKS: list[dict] = [
    {
        "name": "Total Mortality",
        "prepare_func": prepare_cox_data,
        "duration_col": "Follow Up Data",
        "event_col": "Total mortality",
    },
    {
        "name": "CV Mortality",
        "prepare_func": prepare_cardiovascular_data,
        "duration_col": "Follow Up Data",
        "event_col": "death",
    },
    {
        "name": "Myocardial Infarction",
        "prepare_func": prepare_mi_data,
        "duration_col": "MI_date",
        "event_col": "events",
    },
    {
        "name": "Stroke",
        "prepare_func": prepare_stroke_data,
        "duration_col": "Ictus_date",
        "event_col": "events",
    },
]


def default_model_factories() -> dict[str, Callable[[], Any]]:
    """Return a fresh set of default model factories.

    Each value is a zero-argument callable that returns an unfitted model
    instance.  Using factories (rather than shared instances) prevents state
    leaking between folds.
    """
    return {
        "CoxPH": CoxPHWrapper,
        "DeepSurv": DeepSurvWrapper,
        "DeepHit": DeepHitWrapper,
        "RSF": RSFWrapper,
        "GBSA": GBSAWrapper,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_metric_comparison(
    df_results: pd.DataFrame,
    metric: str,
    out_path: str,
    title: Optional[str] = None,
) -> None:
    """Bar chart of per-task, per-model metric with standard-deviation error bars."""
    df_plot = df_results.dropna(subset=[metric])
    if df_plot.empty:
        return

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df_plot,
        x="Task",
        y=metric,
        hue="Model",
        capsize=0.05,
        errorbar="sd",
    )
    plt.title(title or f"Model Comparison — {metric}")
    plt.ylim(
        max(0.0, float(df_plot[metric].min()) - 0.05),
        min(1.0, float(df_plot[metric].max()) + 0.05),
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def _plot_heatmap(
    summary_df: pd.DataFrame,
    metric: str,
    out_path: str,
) -> None:
    """Heatmap of mean metric values (Tasks × Models)."""
    try:
        pivot = summary_df.xs(metric, axis=1, level=0)["mean"].unstack("Model")
    except KeyError:
        return

    plt.figure(figsize=(max(8, len(pivot.columns)), max(4, len(pivot))))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        vmin=0.5,
        vmax=1.0,
    )
    plt.title(f"Mean {metric} (Task × Model)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_evaluation_pipeline(
    data_dir: str = "Dataset Sirbu",
    results_dir: str = "results",
    n_splits: int = 5,
    random_state: int = 42,
    tasks: Optional[list[dict]] = None,
    model_factories: Optional[dict[str, Callable[[], Any]]] = None,
    verbose: bool = False,
    reference_model: str = "CoxPH",
) -> dict[str, pd.DataFrame]:
    """Run the full evaluation pipeline.

    Parameters
    ----------
    data_dir:
        Path to the directory containing the Sirbu dataset Excel files.
    results_dir:
        Directory where CSVs and PDF plots are saved.
    n_splits:
        Number of CV folds.
    random_state:
        Global random seed.
    tasks:
        Override the default task list.  Each dict must have keys
        ``"name"``, ``"prepare_func"``, ``"duration_col"``, ``"event_col"``.
    model_factories:
        Override the default model factory dict.
    verbose:
        Forward verbose flag to training loop.
    reference_model:
        Baseline for Wilcoxon statistical tests.

    Returns
    -------
    dict with keys:
        ``"fold_results"``      — fold-level metrics DataFrame
        ``"summary"``           — mean ± std per (Task, Model)
        ``"statistical_tests"`` — Wilcoxon comparison vs. reference model
    """
    os.makedirs(results_dir, exist_ok=True)

    if tasks is None:
        tasks = DEFAULT_TASKS
    if model_factories is None:
        model_factories = default_model_factories()

    # -------------------------------------------------------------------------
    # Step 1 & 2 — Load and clean data
    # -------------------------------------------------------------------------
    print("Loading data ...")
    df_raw = load_and_merge_data(data_dir)
    print("Cleaning and imputing ...")
    df_clean = clean_and_impute(df_raw)
    print(f"Dataset shape after cleaning: {df_clean.shape}")

    # -------------------------------------------------------------------------
    # Step 3 — Cross-validate each task
    # -------------------------------------------------------------------------
    all_fold_rows: list[pd.DataFrame] = []

    for task in tasks:
        task_name: str = task["name"]
        prepare_func: Callable = task["prepare_func"]
        duration_col: str = task["duration_col"]
        event_col: str = task["event_col"]

        print(f"\n{'='*55}")
        print(f"TASK: {task_name}")
        print(f"{'='*55}")

        # Sanity-check: can we build a feature matrix at all?
        try:
            df_check, _ = prepare_func(df_clean.copy(), scaler=None)
            n_features = df_check.shape[1] - 2  # minus duration + event
            n_events = int((df_check[event_col] > 0).sum())
            print(
                f"  Features: {n_features}, Events: {n_events} / {len(df_check)}"
                f" ({100*n_events/len(df_check):.1f}%)"
            )
        except Exception as exc:
            print(f"  Skipping task '{task_name}': {exc}")
            continue

        # Fresh models for this task
        task_models = {name: factory() for name, factory in model_factories.items()}

        task_results = cross_validate(
            df=df_clean,
            duration_col=duration_col,
            event_col=event_col,
            models=task_models,
            n_splits=n_splits,
            random_state=random_state,
            prepare_func=prepare_func,
            verbose=verbose,
            task_name=task_name,
        )
        all_fold_rows.append(task_results)

    if not all_fold_rows:
        print("No tasks completed successfully.")
        return {}

    fold_results = pd.concat(all_fold_rows, ignore_index=True)
    fold_results.to_csv(os.path.join(results_dir, "fold_results.csv"), index=False)
    print(f"\nFold-level results saved to {results_dir}/fold_results.csv")

    # -------------------------------------------------------------------------
    # Step 4 — Aggregate and summarise
    # -------------------------------------------------------------------------
    summary = summarise_results(fold_results)
    summary.to_csv(os.path.join(results_dir, "summary.csv"))
    print(f"Summary saved to {results_dir}/summary.csv")
    print("\n" + str(summary))

    # -------------------------------------------------------------------------
    # Step 5 — Statistical tests
    # -------------------------------------------------------------------------
    stats_df = run_statistical_tests(
        fold_results, metric="C-index", reference_model=reference_model
    )
    stats_path = os.path.join(results_dir, "statistical_tests.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistical test results saved to {stats_path}")
    print(stats_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Step 6 — Visualisations
    # -------------------------------------------------------------------------
    for metric in ["C-index", "IBS", "AUC_mean"]:
        if metric not in fold_results.columns:
            continue
        _plot_metric_comparison(
            fold_results,
            metric=metric,
            out_path=os.path.join(results_dir, f"comparison_{metric.lower().replace('-', '_')}.pdf"),
            title=f"Model Comparison — {metric}",
        )

    if not summary.empty:
        for metric in ["C-index", "IBS"]:
            try:
                _plot_heatmap(
                    summary,
                    metric=metric,
                    out_path=os.path.join(results_dir, f"heatmap_{metric.lower().replace('-', '_')}.pdf"),
                )
            except Exception:
                pass

    print(f"\nAll outputs saved in: {os.path.abspath(results_dir)}")

    return {
        "fold_results": fold_results,
        "summary": summary,
        "statistical_tests": stats_df,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point (called by ``python -m src.evaluate``)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Survival Analysis Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="Dataset Sirbu", help="Path to data directory")
    parser.add_argument("--results-dir", default="results", help="Output directory")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose training output")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["CoxPH", "DeepSurv", "RSF"],
        choices=["CoxPH", "DeepSurv", "DeepHit", "RSF", "GBSA"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["Total Mortality", "CV Mortality"],
        choices=["Total Mortality", "CV Mortality", "Myocardial Infarction", "Stroke"],
        help="Tasks to evaluate",
    )
    args = parser.parse_args()

    all_factories = default_model_factories()
    selected_factories = {k: v for k, v in all_factories.items() if k in args.models}

    all_tasks_map = {t["name"]: t for t in DEFAULT_TASKS}
    selected_tasks = [all_tasks_map[n] for n in args.tasks if n in all_tasks_map]

    run_evaluation_pipeline(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        n_splits=args.folds,
        random_state=args.seed,
        tasks=selected_tasks,
        model_factories=selected_factories,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
