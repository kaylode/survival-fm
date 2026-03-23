"""
scripts/benchmark.py — External benchmark dataset runner for the SurvPFN project.

Loads SUPPORT2, METABRIC, and GBSG datasets, runs 5-fold CV with all models
(including TabPFN embedding variants), and saves results to
``results/benchmark_<dataset>.csv``.

CLI
---
    uv run python scripts/benchmark.py --datasets SUPPORT2 METABRIC GBSG --models all
    uv run python scripts/benchmark.py --datasets METABRIC --models cox rsf gbsa deepsurv mtlr
    uv run python scripts/benchmark.py --datasets GBSG --tune --trials 20

Available model names (for --models)
-------------------------------------
  all, cox, rsf, gbsa, deepsurv, mtlr, pchazard, deephit_single, embedding_cox

Notes
-----
The TabPFN embedding model (embedding_cox) requires the ``tabpfn`` package:
    uv add tabpfn
All other models require:
    uv add pycox torchtuples scikit-survival optuna sksurv
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from survpfn.data.benchmarks import BENCHMARK_DATASETS
from survpfn.eval.metrics import evaluate_survival_model

# Lazy imports: model modules are only imported when needed to avoid hard
# dependency failures when a model is not selected.


def _make_df(X: np.ndarray, y_time: np.ndarray, y_event: np.ndarray) -> pd.DataFrame:
    """Combine feature matrix and labels into a single DataFrame."""
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["duration"] = y_time
    df["event"] = y_event
    return df


DURATION_COL = "duration"
EVENT_COL = "event"


def run_cox(df_train, df_test, tune, n_trials):
    from survpfn.models.classical import run_multivariate_cox
    _, _, _, c_index = run_multivariate_cox(df_train, df_test, DURATION_COL, EVENT_COL)
    return {"C-index": c_index}


def run_rsf(df_train, df_test, tune, n_trials):
    from survpfn.models.tree import train_rsf
    from survpfn.eval.metrics import evaluate_survival_model
    _, risk, surv_p, surv_t = train_rsf(df_train, df_test, DURATION_COL, EVENT_COL,
                                         tune=tune, n_trials=n_trials)
    return evaluate_survival_model(df_train, df_test, DURATION_COL, EVENT_COL,
                                   risk, surv_probs=surv_p, surv_times=surv_t)


def run_gbsa(df_train, df_test, tune, n_trials):
    from survpfn.models.tree import train_gbsa
    _, risk, surv_p, surv_t = train_gbsa(df_train, df_test, DURATION_COL, EVENT_COL,
                                          tune=tune, n_trials=n_trials)
    return evaluate_survival_model(df_train, df_test, DURATION_COL, EVENT_COL,
                                   risk, surv_probs=surv_p, surv_times=surv_t)


def run_deepsurv(df_train, df_test, tune, n_trials):
    from survpfn.models.deep_surv import train_deepsurv
    _, risk, surv_p, surv_t = train_deepsurv(df_train, df_test, DURATION_COL, EVENT_COL,
                                              tune=tune, n_trials=n_trials)
    return evaluate_survival_model(df_train, df_test, DURATION_COL, EVENT_COL,
                                   risk, surv_probs=surv_p, surv_times=surv_t)


def run_mtlr(df_train, df_test, tune, n_trials):
    from survpfn.models.deep_hit import train_mtlr
    _, risk, surv_p, surv_t = train_mtlr(df_train, df_test, DURATION_COL, EVENT_COL,
                                          tune=tune, n_trials=n_trials)
    return evaluate_survival_model(df_train, df_test, DURATION_COL, EVENT_COL,
                                   risk, surv_probs=surv_p, surv_times=surv_t)


def run_pchazard(df_train, df_test, tune, n_trials):
    from survpfn.models.deep_hit import train_pchazard
    _, risk, surv_p, surv_t = train_pchazard(df_train, df_test, DURATION_COL, EVENT_COL,
                                              tune=tune, n_trials=n_trials)
    return evaluate_survival_model(df_train, df_test, DURATION_COL, EVENT_COL,
                                   risk, surv_probs=surv_p, surv_times=surv_t)


def run_deephit_single(df_train, df_test, tune, n_trials):
    from survpfn.models.deep_hit import train_deephit_single
    _, risk, surv_p, surv_t = train_deephit_single(df_train, df_test, DURATION_COL, EVENT_COL,
                                                    tune=tune, n_trials=n_trials)
    return evaluate_survival_model(df_train, df_test, DURATION_COL, EVENT_COL,
                                   risk, surv_probs=surv_p, surv_times=surv_t)


def run_embedding_cox(df_train, df_test, tune, n_trials):
    from survpfn.models.tabpfn.cox import train_embedding_cox
    _, risk, surv_p, surv_t = train_embedding_cox(df_train, df_test, DURATION_COL, EVENT_COL,
                                                   tune=tune, n_trials=n_trials)
    return evaluate_survival_model(df_train, df_test, DURATION_COL, EVENT_COL,
                                   risk, surv_probs=surv_p, surv_times=surv_t)


ALL_MODELS: dict[str, Callable] = {
    "cox": run_cox,
    "rsf": run_rsf,
    "gbsa": run_gbsa,
    "deepsurv": run_deepsurv,
    "mtlr": run_mtlr,
    "pchazard": run_pchazard,
    "deephit_single": run_deephit_single,
    "embedding_cox": run_embedding_cox,
}


def run_benchmark(
    dataset_name: str,
    model_names: list[str],
    n_folds: int = 5,
    tune: bool = False,
    n_trials: int = 10,
    output_dir: str = "results",
    random_state: int = 42,
) -> pd.DataFrame:
    """Run all specified models on one benchmark dataset with k-fold CV.

    Parameters
    ----------
    dataset_name:
        One of "SUPPORT2", "METABRIC", "GBSG".
    model_names:
        List of model keys from ``ALL_MODELS``.
    n_folds:
        Number of cross-validation folds.
    tune:
        Whether to enable Optuna hyperparameter tuning.
    n_trials:
        Number of Optuna trials per model per fold.
    output_dir:
        Directory for saving CSV results.
    random_state:
        Seed for StratifiedKFold.

    Returns
    -------
    pd.DataFrame with fold-level results.
    """
    print(f"\n{'='*50}")
    print(f"BENCHMARK DATASET: {dataset_name}")
    print(f"{'='*50}")

    loader = BENCHMARK_DATASETS[dataset_name]
    X, y_time, y_event = loader()
    print(f"Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features, "
          f"event rate={y_event.mean():.3f}")

    # Use binarized event for stratification (handles edge case of all-zero or float)
    strat_label = (y_event > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, strat_label)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_time_train, y_time_test = y_time[train_idx], y_time[test_idx]
        y_event_train, y_event_test = y_event[train_idx], y_event[test_idx]

        # Fold-level StandardScaler (fitted only on train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        df_train = _make_df(X_train_scaled, y_time_train, y_event_train)
        df_test = _make_df(X_test_scaled, y_time_test, y_event_test)

        for model_name in model_names:
            print(f"  Model: {model_name}", end=" ... ", flush=True)
            model_fn = ALL_MODELS[model_name]
            try:
                metrics = model_fn(df_train, df_test, tune, n_trials)
                row = {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Fold": fold + 1,
                    **metrics,
                }
            except Exception as exc:
                warnings.warn(f"Model {model_name} failed on fold {fold + 1}: {exc}", stacklevel=2)
                row = {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Fold": fold + 1,
                    "C-index": float("nan"),
                }
            print(f"C-index={row.get('C-index', float('nan')):.4f}" if not np.isnan(row.get('C-index', float('nan'))) else "FAILED")
            results.append(row)

    df_results = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"benchmark_{dataset_name.lower()}.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    # Summary
    summary = (
        df_results.groupby("Model")["C-index"]
        .agg(["mean", "std"])
        .round(4)
        .rename(columns={"mean": "C-index_mean", "std": "C-index_std"})
        .sort_values("C-index_mean", ascending=False)
    )
    print(f"\nSummary for {dataset_name}:")
    print(summary.to_string())

    return df_results


def main():
    parser = argparse.ArgumentParser(
        description="SurvPFN external benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SUPPORT2", "METABRIC", "GBSG"],
        choices=list(BENCHMARK_DATASETS.keys()),
        help="Benchmark datasets to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=(
            "Models to evaluate. Use 'all' or specify any of: "
            + ", ".join(ALL_MODELS.keys())
        ),
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable Optuna hyperparameter tuning.",
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of Optuna trials per model per fold.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for StratifiedKFold.",
    )
    args = parser.parse_args()

    # Resolve model list
    if "all" in args.models:
        model_names = list(ALL_MODELS.keys())
    else:
        model_names = []
        for m in args.models:
            if m not in ALL_MODELS:
                parser.error(
                    f"Unknown model '{m}'. Choose from: all, "
                    + ", ".join(ALL_MODELS.keys())
                )
            model_names.append(m)

    all_results = []
    for dataset in args.datasets:
        df = run_benchmark(
            dataset_name=dataset,
            model_names=model_names,
            n_folds=args.folds,
            tune=args.tune,
            n_trials=args.trials,
            output_dir=args.output_dir,
            random_state=args.seed,
        )
        all_results.append(df)

    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "benchmark_all.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
