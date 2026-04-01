"""survpfn.eval.metrics — survival model evaluation metrics."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
    cumulative_dynamic_auc
)
from scipy.stats import wilcoxon

def _get_survival_array(y, duration_col, event_col):
    """Convert pandas dataframe/series into structured array for sksurv."""
    return np.array(
        list(zip(y[event_col].astype(bool), y[duration_col])),
        dtype=[('event', '?'), ('time', '<f8')]
    )


def _safe_time_grid(
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_points: int = 50,
) -> np.ndarray:
    """Return a uniform time grid within the shared support of train and test."""
    lo = max(float(y_train["time"].min()), float(y_test["time"].min()))
    hi = min(float(y_train["time"].max()), float(y_test["time"].max()))
    if lo >= hi:
        return np.array([])
    # Use a relative margin (0.1% of the range) to avoid precision issues at boundaries.
    margin = (hi - lo) * 0.001
    return np.linspace(lo + margin, hi - margin, n_points)


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
    For each uncensored subject i:
    1. Interpolate S_i(t_i) — the predicted survival probability at the
       observed event time.
    2. Collect all S_i(t_i) values.
    3. Check that the resulting distribution is approximately Uniform(0, 1)
       by computing the chi-squared statistic against a uniform histogram.

    Parameters
    ----------
    events:
        Boolean array of event indicators.
    times:
        Time-to-event or censoring times (same length as events).
    surv_probs:
        Predicted survival matrix (n_samples, n_surv_times).
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


def evaluate_survival_model(
    y_train,
    y_test,
    duration_col,
    event_col,
    risk_scores,
    surv_probs=None,
    surv_times=None,
    time_horizons=None,
    n_time_points: int = 50,
):
    """
    Evaluate survival metrics.
    Args:
        y_train: Training labels df (contains duration and event).
        y_test: Test labels df.
        duration_col: Name of time column.
        event_col: Name of event column.
        risk_scores: Array of risk scores (e.g. hazard from Cox or -surv from others).
        surv_probs: Matrix of survival probabilities shape (n_samples, n_times).
        surv_times: Array of times corresponding to columns of surv_probs.
        time_horizons: Specific times to evaluate time-dependent AUC and IBS.
        n_time_points: Number of evenly spaced points for the IBS integral.
    Returns:
        Dictionary of metrics {"C-index": ..., "IBS": ..., "AUC_mean": ..., "D-cal": ...}
    """
    y_train_sksurv = _get_survival_array(y_train, duration_col, event_col)
    y_test_sksurv = _get_survival_array(y_test, duration_col, event_col)

    metrics = {}

    # 1. Concordance Index
    # Some models predict survival (higher is better), risk scores must be higher for worse outcomes.
    try:
        c_index, _, _, _, _ = concordance_index_censored(
            y_test_sksurv["event"], y_test_sksurv["time"], risk_scores
        )
        metrics["C-index"] = c_index
    except Exception as e:
        warnings.warn(f"C-index computation failed: {e}", stacklevel=2)
        metrics["C-index"] = np.nan

    if surv_probs is not None and surv_times is not None:
        grid = _safe_time_grid(y_train_sksurv, y_test_sksurv, n_points=n_time_points)
        if len(grid) == 0:
            return metrics

        if time_horizons is None:
            # Pick a few percentiles from the test set event times
            mask = y_test_sksurv["event"]
            if sum(mask) > 0:
                time_horizons = np.percentile(y_test_sksurv["time"][mask], [25, 50, 75])
            else:
                time_horizons = grid[[len(grid) // 4, len(grid) // 2, 3 * len(grid) // 4]]

        lo, hi = float(grid[0]), float(grid[-1])
        valid_times = np.array([t for t in time_horizons if lo < t < hi])

        if len(valid_times) > 0:
            try:
                # 2. Integrated Brier Score — use full grid for proper integration
                surv_at_grid = np.zeros((surv_probs.shape[0], len(grid)))
                for i in range(surv_probs.shape[0]):
                    surv_at_grid[i, :] = np.interp(grid, surv_times, surv_probs[i, :])

                ibs = integrated_brier_score(
                    y_train_sksurv, y_test_sksurv, surv_at_grid, grid
                )
                metrics["IBS"] = ibs

                # 3. Time-dependent AUC
                auc, mean_auc = cumulative_dynamic_auc(
                    y_train_sksurv, y_test_sksurv, risk_scores, valid_times
                )
                metrics["AUC_mean"] = mean_auc

                for t, a in zip(valid_times, auc):
                    metrics[f"AUC_t={t:.1f}"] = a
            except Exception as e:
                # In case some metrics fail (e.g. no events before T)
                warnings.warn(f"Time-dependent metrics computation failed: {e}", stacklevel=2)

        # 4. D-calibration (reliability)
        try:
            metrics["D-cal"] = d_calibration(
                y_test_sksurv["event"], y_test_sksurv["time"],
                surv_probs, surv_times, n_bins=10
            )
        except Exception as e:
            warnings.warn(f"D-calibration computation failed: {e}", stacklevel=2)
            metrics["D-cal"] = float("nan")

    return metrics

def run_statistical_tests(results_df, metric="C-index", reference_model="Multivariate Cox"):
    """
    Run Wilcoxon signed-rank test comparing reference_model to all other models.
    """
    stats_res = []

    for task in results_df["Task"].unique():
        task_df = results_df[results_df["Task"] == task]
        ref_df = task_df[task_df["Model"] == reference_model]
        if ref_df.empty:
            continue

        ref_scores = ref_df.sort_values("Fold")[metric].values

        for model in task_df["Model"].unique():
            if model == reference_model:
                continue

            mod_df = task_df[task_df["Model"] == model].sort_values("Fold")
            mod_scores = mod_df[metric].values

            if len(mod_scores) == len(ref_scores) and len(ref_scores) > 1:
                # Differences
                diff = mod_scores - ref_scores
                if np.all(diff == 0):
                    p_val = 1.0
                else:
                    try:
                        _, p_val = wilcoxon(mod_scores, ref_scores, zero_method='zsplit')
                    except Exception:
                        p_val = np.nan

                stats_res.append({
                    "Task": task,
                    "Reference": reference_model,
                    "Model": model,
                    "Mean_Diff": float(np.mean(diff)),
                    "Median_Diff": float(np.median(diff)),
                    "p-value": p_val,
                    "Significant": bool(p_val < 0.05) if not np.isnan(p_val) else False
                })

    return pd.DataFrame(stats_res)
