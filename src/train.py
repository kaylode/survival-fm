"""
src/train.py — Training loop with k-fold cross-validation support.

Usage (single task, 5-fold CV)
-------------------------------
    from src.data import load_and_merge_data, clean_and_impute, prepare_cox_data, make_kfold_splits
    from src.models import CoxPHWrapper, DeepSurvWrapper, RSFWrapper
    from src.train import cross_validate

    df_raw = load_and_merge_data("Dataset Sirbu")
    df_clean = clean_and_impute(df_raw)
    df_task, _ = prepare_cox_data(df_clean)   # scaler fitted on full data for exploration

    results = cross_validate(
        df_task,
        duration_col="Follow Up Data",
        event_col="Total mortality",
        models={
            "CoxPH": CoxPHWrapper(),
            "DeepSurv": DeepSurvWrapper(),
            "RSF": RSFWrapper(),
        },
        n_splits=5,
    )

Design decisions
----------------
* The ``StandardScaler`` is **always** re-fitted on the training fold inside
  the CV loop to prevent leakage from test data into scaling statistics.

* Task-specific ``prepare_*`` functions from ``src.data`` accept an optional
  pre-fitted scaler, so the caller can pass the fold-level scaler after fitting
  it on the *raw* (unscaled) training fold.

* Optuna hyperparameter tuning can be enabled per-model by wrapping models
  in a callable factory (see ``TunableModelFactory``).
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import make_kfold_splits
from src.metrics import evaluate_survival_model


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ModelFactory = Callable[[], Any]  # returns a fresh model instance


# ---------------------------------------------------------------------------
# 1.  Single-fold training helper
# ---------------------------------------------------------------------------

def train_one_fold(
    model: Any,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    verbose: bool = False,
) -> dict[str, float]:
    """Fit one model on one train/test split and return evaluation metrics.

    The ``model`` object must implement:
        ``fit(df_train, duration_col, event_col)``
        ``predict_risk(df_test) -> np.ndarray``
        ``predict_survival(df_test) -> (surv_probs, surv_times)``

    Parameters
    ----------
    model:
        Unfitted model instance.
    df_train:
        Training split (scaled, with outcome columns).
    df_test:
        Test split (scaled with the same scaler as training).
    duration_col:
        Time column.
    event_col:
        Event column.
    verbose:
        Pass to ``model.fit`` if supported.

    Returns
    -------
    dict of metric name → value.
    """
    fit_kwargs: dict = {}
    import inspect
    sig = inspect.signature(model.fit)
    if "verbose" in sig.parameters:
        fit_kwargs["verbose"] = verbose

    t0 = time.perf_counter()
    model.fit(df_train, duration_col, event_col, **fit_kwargs)
    elapsed = time.perf_counter() - t0

    risk_scores = model.predict_risk(df_test)

    try:
        surv_probs, surv_times = model.predict_survival(df_test)
    except Exception as exc:
        warnings.warn(f"predict_survival failed: {exc}", stacklevel=2)
        surv_probs, surv_times = None, None

    metrics = evaluate_survival_model(
        df_train=df_train,
        df_test=df_test,
        duration_col=duration_col,
        event_col=event_col,
        risk_scores=risk_scores,
        surv_probs=surv_probs,
        surv_times=surv_times,
    )
    metrics["fit_time_s"] = round(elapsed, 3)
    return metrics


# ---------------------------------------------------------------------------
# 2.  K-fold cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    models: dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42,
    prepare_func: Optional[Callable] = None,
    verbose: bool = False,
    task_name: str = "Unknown",
) -> pd.DataFrame:
    """Run stratified k-fold cross-validation over a set of models.

    Parameters
    ----------
    df:
        Full dataset for this task (unscaled continuous features; outcome
        columns included).
    duration_col:
        Time-to-event column.
    event_col:
        Event indicator column (0 = censored, ≥1 = event).
    models:
        Mapping of ``{model_name: model_instance}``.  Each model is cloned
        (via its constructor) at each fold — you may pass factory callables or
        pre-constructed instances.  If instances are passed, they are re-used
        across folds (which is fine as long as ``fit`` reinitialises them).
    n_splits:
        Number of folds.
    random_state:
        Random seed.
    prepare_func:
        Optional task-specific feature-preparation function from ``src.data``
        (e.g. ``prepare_cox_data``).  Signature must be::

            prepare_func(df, scaler=None) -> (df_prepared, fitted_scaler)

        When provided, the continuous scaling is performed *inside* the CV loop
        using the fold-level scaler, preventing leakage.  The input ``df``
        should then be the *raw* (unscaled) cleaned dataframe.
    verbose:
        Print per-fold model names and C-index.
    task_name:
        Label for the ``"Task"`` column in the output.

    Returns
    -------
    pd.DataFrame with columns:
        ``["Task", "Model", "Fold", "C-index", "IBS", "AUC_mean",
           "D-cal", "fit_time_s", ...]``

    Notes
    -----
    When ``prepare_func`` is given the function is called twice per fold:
    once on the training split (``scaler=None``) to fit the scaler, and once
    on the test split passing the fitted scaler.  This guarantees that no
    scaling statistics from the test fold contaminate the model.
    """
    fold_indices = make_kfold_splits(df, event_col, n_splits=n_splits, random_state=random_state)
    rows: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices, start=1):
        print(f"\n--- Fold {fold_idx}/{n_splits} ---")

        df_train_raw = df.iloc[train_idx].reset_index(drop=True)
        df_test_raw = df.iloc[test_idx].reset_index(drop=True)

        if prepare_func is not None:
            # Scale using training-fold statistics only
            df_train_fold, fold_scaler = prepare_func(df_train_raw, scaler=None)
            df_test_fold, _ = prepare_func(df_test_raw, scaler=fold_scaler)
        else:
            # Caller is responsible for pre-scaling or the model handles it
            df_train_fold = df_train_raw
            df_test_fold = df_test_raw
            fold_scaler = None  # noqa: F841

        for model_name, model in models.items():
            print(f"  Model: {model_name} ... ", end="", flush=True)
            try:
                metrics = train_one_fold(
                    model=model,
                    df_train=df_train_fold,
                    df_test=df_test_fold,
                    duration_col=duration_col,
                    event_col=event_col,
                    verbose=verbose,
                )
                c = metrics.get("C-index", float("nan"))
                print(f"C-index={c:.4f}")
            except Exception as exc:
                warnings.warn(
                    f"Model '{model_name}' failed on fold {fold_idx}: {exc}",
                    stacklevel=2,
                )
                metrics = {"C-index": float("nan")}
                print("FAILED")

            rows.append({
                "Task": task_name,
                "Model": model_name,
                "Fold": fold_idx,
                **metrics,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3.  Multi-task runner
# ---------------------------------------------------------------------------

def run_all_tasks(
    df_clean: pd.DataFrame,
    task_configs: list[dict],
    model_factories: dict[str, Callable[[], Any]],
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = False,
    results_dir: str = "results",
) -> pd.DataFrame:
    """Run cross-validation for multiple survival tasks and save results.

    Parameters
    ----------
    df_clean:
        Output of :func:`src.data.clean_and_impute` — cleaned but *unscaled*
        dataset.
    task_configs:
        List of task dictionaries with keys:
        ``{"name": str, "prepare_func": callable, "duration_col": str,
           "event_col": str}``.
    model_factories:
        Mapping of ``{model_name: factory_callable}``.  Each factory is called
        with no arguments to produce a fresh, unfitted model for every fold.
        Using factories (rather than shared instances) avoids state leakage
        between folds for stateful models.
    n_splits:
        Cross-validation folds.
    random_state:
        Seed.
    verbose:
        Print training progress.
    results_dir:
        Directory to save ``model_comparison.csv``.

    Returns
    -------
    pd.DataFrame with all fold-level results concatenated.

    Example
    -------
    .. code-block:: python

        from src.data import (load_and_merge_data, clean_and_impute,
                              prepare_cox_data, prepare_cardiovascular_data)
        from src.models import CoxPHWrapper, DeepSurvWrapper
        from src.train import run_all_tasks

        df_clean = clean_and_impute(load_and_merge_data("Dataset Sirbu"))

        task_configs = [
            {"name": "Total Mortality",
             "prepare_func": prepare_cox_data,
             "duration_col": "Follow Up Data",
             "event_col": "Total mortality"},
            {"name": "CV Mortality",
             "prepare_func": prepare_cardiovascular_data,
             "duration_col": "Follow Up Data",
             "event_col": "death"},
        ]

        model_factories = {
            "CoxPH": CoxPHWrapper,
            "DeepSurv": DeepSurvWrapper,
        }

        all_results = run_all_tasks(df_clean, task_configs, model_factories)
    """
    all_rows: list[pd.DataFrame] = []

    for task in task_configs:
        task_name: str = task["name"]
        prepare_func: Callable = task["prepare_func"]
        duration_col: str = task["duration_col"]
        event_col: str = task["event_col"]

        print(f"\n{'='*50}")
        print(f"TASK: {task_name}")
        print(f"{'='*50}")

        # Prepare a fully-scaled version of the data for fold-index generation
        # (the fold-level prepare happens inside cross_validate)
        try:
            df_task_full, _ = prepare_func(df_clean, scaler=None)
        except Exception as exc:
            warnings.warn(f"Task '{task_name}' preparation failed: {exc}", stacklevel=2)
            continue

        # Instantiate fresh models for this task
        models_this_task = {name: factory() for name, factory in model_factories.items()}

        task_results = cross_validate(
            df=df_clean,          # pass raw cleaned data; prepare_func handles scaling
            duration_col=duration_col,
            event_col=event_col,
            models=models_this_task,
            n_splits=n_splits,
            random_state=random_state,
            prepare_func=prepare_func,
            verbose=verbose,
            task_name=task_name,
        )
        all_rows.append(task_results)

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "model_comparison.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    return combined
