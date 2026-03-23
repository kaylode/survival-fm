"""
src/metrics.py — Survival-analysis evaluation metrics.

Metrics provided
----------------
* **C-index** (Harrell / Antolini) via ``scikit-survival``
* **Integrated Brier Score (IBS)** via ``scikit-survival``
* **Time-dependent AUROC** via ``scikit-survival``
* **D-calibration** (reliability) via binned observed vs. expected survival
* **Competing-risk C-index** per cause via cause-specific approach
* **Wilcoxon signed-rank statistical tests** for fold-level comparisons

All functions follow a consistent signature pattern and return plain dicts or
DataFrames so results can be written directly to CSV.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _structured_array(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    binary_event: bool = True,
) -> np.ndarray:
    """Convert a DataFrame to a scikit-survival structured array.

    Parameters
    ----------
    df:
        DataFrame containing at least ``duration_col`` and ``event_col``.
    duration_col:
        Column with time-to-event or censoring time (numeric, days).
    event_col:
        Column with the event indicator.  If ``binary_event=True``, any
        non-zero value is treated as an event.
    binary_event:
        When ``True`` the event column is cast to bool (handles competing-risk
        codes where event ∈ {0, 1, 2}).

    Returns
    -------
    np.ndarray with dtype ``[('event', '?'), ('time', '<f8')]``
    """
    event = df[event_col].astype(bool) if binary_event else df[event_col]
    return np.array(
        list(zip(event, df[duration_col])),
        dtype=[("event", "?"), ("time", "<f8")],
    )


def _safe_time_grid(
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_points: int = 50,
) -> np.ndarray:
    """Return a uniform time grid within the *shared* support of train and test.

    Using times outside the training support leads to extrapolation; using
    times beyond the last test event raises errors in ``sksurv`` metrics.
    """
    lo = max(float(y_train["time"].min()), float(y_test["time"].min()))
    hi = min(float(y_train["time"].max()), float(y_test["time"].max()))
    if lo >= hi:
        return np.array([])
    return np.linspace(lo + 1e-6, hi - 1e-6, n_points)


# ---------------------------------------------------------------------------
# 1.  Primary evaluation function
# ---------------------------------------------------------------------------

def evaluate_survival_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    risk_scores: np.ndarray,
    surv_probs: Optional[np.ndarray] = None,
    surv_times: Optional[np.ndarray] = None,
    time_horizons: Optional[np.ndarray] = None,
    n_time_points: int = 50,
) -> dict[str, float]:
    """Evaluate a survival model on the test set.

    Parameters
    ----------
    df_train:
        Training split (used to compute IBS and AUC baselines / training
        survival distribution).
    df_test:
        Test split.
    duration_col:
        Name of the time-to-event column.
    event_col:
        Name of the event indicator column (0 = censored, ≥1 = event).
    risk_scores:
        1-D array of per-subject risk scores (higher = worse prognosis).
    surv_probs:
        Optional array of shape ``(n_test_samples, n_times)`` with predicted
        survival probabilities at ``surv_times``.  Required for IBS and AUC.
    surv_times:
        1-D array of time points corresponding to columns of ``surv_probs``.
    time_horizons:
        Specific times at which time-dependent AUC is evaluated. Defaults to
        the 25th, 50th, and 75th percentile of observed event times in the
        test set.
    n_time_points:
        Number of evenly spaced points for the IBS integral when
        ``time_horizons`` is not provided.

    Returns
    -------
    dict with keys: ``"C-index"``, ``"IBS"``, ``"AUC_mean"``,
    ``"AUC_t=<t>"`` for each horizon, and ``"D-cal"`` (lower = better).
    """
    metrics: dict[str, float] = {}

    y_train_sa = _structured_array(df_train, duration_col, event_col)
    y_test_sa = _structured_array(df_test, duration_col, event_col)

    # ---- 1a. Harrell / Antolini C-index ------------------------------------
    try:
        c, _, _, _, _ = concordance_index_censored(
            y_test_sa["event"], y_test_sa["time"], risk_scores
        )
        metrics["C-index"] = float(c)
    except Exception as exc:
        warnings.warn(f"C-index computation failed: {exc}", stacklevel=2)
        metrics["C-index"] = float("nan")

    # ---- 1b. Survival-probability–based metrics ----------------------------
    if surv_probs is None or surv_times is None:
        return metrics

    grid = _safe_time_grid(y_train_sa, y_test_sa, n_points=n_time_points)
    if len(grid) == 0:
        return metrics

    if time_horizons is None:
        event_mask = y_test_sa["event"]
        if event_mask.sum() > 0:
            time_horizons = np.percentile(y_test_sa["time"][event_mask], [25, 50, 75])
        else:
            time_horizons = grid[[len(grid) // 4, len(grid) // 2, 3 * len(grid) // 4]]

    # Restrict horizons to the shared support
    lo, hi = float(grid[0]), float(grid[-1])
    valid_times = np.asarray([t for t in time_horizons if lo < t < hi])
    if len(valid_times) == 0:
        return metrics

    # Interpolate survival probabilities onto evaluation times
    surv_at_valid = np.zeros((surv_probs.shape[0], len(valid_times)))
    for i in range(surv_probs.shape[0]):
        surv_at_valid[i, :] = np.interp(valid_times, surv_times, surv_probs[i, :])

    # ---- 1c. Integrated Brier Score ----------------------------------------
    # Extend evaluation to the full grid for proper integration
    surv_at_grid = np.zeros((surv_probs.shape[0], len(grid)))
    for i in range(surv_probs.shape[0]):
        surv_at_grid[i, :] = np.interp(grid, surv_times, surv_probs[i, :])

    try:
        ibs = integrated_brier_score(y_train_sa, y_test_sa, surv_at_grid, grid)
        metrics["IBS"] = float(ibs)
    except Exception as exc:
        warnings.warn(f"IBS computation failed: {exc}", stacklevel=2)
        metrics["IBS"] = float("nan")

    # ---- 1d. Time-dependent AUROC ------------------------------------------
    try:
        auc_vals, mean_auc = cumulative_dynamic_auc(
            y_train_sa, y_test_sa, risk_scores, valid_times
        )
        metrics["AUC_mean"] = float(mean_auc)
        for t, a in zip(valid_times, auc_vals):
            metrics[f"AUC_t={t:.1f}"] = float(a)
    except Exception as exc:
        warnings.warn(f"Time-dependent AUC computation failed: {exc}", stacklevel=2)
        metrics["AUC_mean"] = float("nan")

    # ---- 1e. D-calibration (reliability) ------------------------------------
    try:
        metrics["D-cal"] = d_calibration(
            y_test_sa["event"], y_test_sa["time"],
            surv_probs, surv_times, n_bins=10
        )
    except Exception as exc:
        warnings.warn(f"D-calibration computation failed: {exc}", stacklevel=2)
        metrics["D-cal"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# 2.  D-Calibration
# ---------------------------------------------------------------------------

def d_calibration(
    events: np.ndarray,
    times: np.ndarray,
    surv_probs: np.ndarray,
    surv_times: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute D-calibration (distribution calibration) for survival models.

    D-calibration measures whether the predicted survival probability at the
    event time is uniformly distributed over [0, 1] for uncensored subjects.
    A well-calibrated model produces a uniform histogram; the returned value
    is the mean squared deviation from uniformity (lower is better).

    Algorithm (Haider et al., 2020)
    --------------------------------
    For each uncensored subject *i*:
    1. Interpolate ``S_i(t_i)`` — the predicted survival probability at the
       observed event time.
    2. Collect all ``S_i(t_i)`` values.
    3. Check that the resulting distribution is approximately Uniform(0, 1)
       by computing the chi-squared statistic against a uniform histogram.

    Parameters
    ----------
    events:
        Boolean array of event indicators.
    times:
        Time-to-event or censoring times (same length as ``events``).
    surv_probs:
        Predicted survival matrix ``(n_samples, n_surv_times)``.
    surv_times:
        Time points for the survival matrix columns.
    n_bins:
        Number of histogram bins for the uniformity test.

    Returns
    -------
    float
        Mean squared deviation from the uniform density (normalised by bin
        count).  0 = perfectly calibrated.
    """
    uncensored_mask = events.astype(bool)
    if uncensored_mask.sum() < 2:
        return float("nan")

    event_times = times[uncensored_mask]
    event_surv = surv_probs[uncensored_mask]

    # Predicted survival at each subject's own event time
    s_at_t = np.array([
        float(np.interp(t, surv_times, s))
        for t, s in zip(event_times, event_surv)
    ])

    # Histogram over [0, 1]
    counts, _ = np.histogram(s_at_t, bins=n_bins, range=(0.0, 1.0))
    expected = len(s_at_t) / n_bins
    deviation = float(np.mean((counts - expected) ** 2) / (expected ** 2))
    return deviation


# ---------------------------------------------------------------------------
# 3.  Competing-risk C-index
# ---------------------------------------------------------------------------

def competing_risk_cindex(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    duration_col: str,
    event_col: str,
    risk_scores_per_cause: dict[int, np.ndarray],
) -> dict[str, float]:
    """Compute cause-specific C-indices for a competing-risks task.

    Each cause is evaluated as a separate binary outcome (1 = cause-*k* event,
    0 = censored or other cause).

    Parameters
    ----------
    df_train:
        Training split (used for ``_structured_array`` consistency checks).
    df_test:
        Test split.
    duration_col:
        Time column.
    event_col:
        Competing-risk event column (0 = censored, 1 = cause A, 2 = cause B).
    risk_scores_per_cause:
        Mapping ``{cause_code: risk_score_array}``.  E.g. ``{1: scores_cv,
        2: scores_other}``.

    Returns
    -------
    dict with keys like ``"C-index_cause_1"``, ``"C-index_cause_2"``.
    """
    results: dict[str, float] = {}
    for cause, scores in risk_scores_per_cause.items():
        cause_event = (df_test[event_col] == cause).astype(bool)
        try:
            c, _, _, _, _ = concordance_index_censored(
                cause_event, df_test[duration_col].values, scores
            )
            results[f"C-index_cause_{cause}"] = float(c)
        except Exception as exc:
            warnings.warn(
                f"Competing-risk C-index for cause {cause} failed: {exc}",
                stacklevel=2,
            )
            results[f"C-index_cause_{cause}"] = float("nan")
    return results


# ---------------------------------------------------------------------------
# 4.  Statistical comparison across folds
# ---------------------------------------------------------------------------

def run_statistical_tests(
    results_df: pd.DataFrame,
    metric: str = "C-index",
    reference_model: str = "Multivariate Cox",
) -> pd.DataFrame:
    """Wilcoxon signed-rank test comparing each model to a reference baseline.

    The test is applied separately per task using the per-fold metric values.

    Parameters
    ----------
    results_df:
        DataFrame with columns ``["Task", "Model", "Fold", metric]`` (and
        optionally more metric columns).
    metric:
        Column name of the evaluation metric to compare.
    reference_model:
        Name of the baseline model used as comparison anchor.

    Returns
    -------
    pd.DataFrame with columns:
    ``["Task", "Reference", "Model", "Mean_Diff", "Median_Diff",
       "p-value", "Significant"]``
    """
    rows: list[dict] = []

    for task in results_df["Task"].unique():
        task_df = results_df[results_df["Task"] == task]
        ref_df = task_df[task_df["Model"] == reference_model].sort_values("Fold")
        if ref_df.empty:
            continue

        ref_scores = ref_df[metric].values

        for model in task_df["Model"].unique():
            if model == reference_model:
                continue

            mod_df = task_df[task_df["Model"] == model].sort_values("Fold")
            mod_scores = mod_df[metric].values

            if len(mod_scores) != len(ref_scores) or len(ref_scores) < 2:
                continue

            diff = mod_scores - ref_scores
            if np.all(diff == 0):
                p_val = 1.0
            else:
                try:
                    _, p_val = wilcoxon(mod_scores, ref_scores, zero_method="zsplit")
                except Exception:
                    p_val = float("nan")

            rows.append({
                "Task": task,
                "Reference": reference_model,
                "Model": model,
                "Mean_Diff": float(np.mean(diff)),
                "Median_Diff": float(np.median(diff)),
                "p-value": p_val,
                "Significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5.  Summary utility
# ---------------------------------------------------------------------------

def summarise_results(results_df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Aggregate fold-level results into mean ± std per (Task, Model).

    Parameters
    ----------
    results_df:
        Raw fold-level results from the training loop.
    metrics:
        Metric column names to aggregate.  If ``None``, all numeric columns
        except ``"Fold"`` are used.

    Returns
    -------
    pd.DataFrame with multi-level columns ``(metric, mean/std)``.
    """
    if metrics is None:
        metrics = [
            c for c in results_df.select_dtypes(include=[np.number]).columns
            if c != "Fold"
        ]

    agg_funcs = {m: ["mean", "std"] for m in metrics if m in results_df.columns}
    summary = (
        results_df.groupby(["Task", "Model"])
        .agg(agg_funcs)
        .round(4)
    )
    return summary
