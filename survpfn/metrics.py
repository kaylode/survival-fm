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

def evaluate_survival_model(y_train, y_test, duration_col, event_col, risk_scores, surv_probs=None, surv_times=None, time_horizons=None):
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
    Returns:
        Dictionary of metrics {"C-index": ..., "IBS": ..., "AUC_mean": ...}
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
        metrics["C-index"] = np.nan
        
    if surv_probs is not None and surv_times is not None:
        # Filter evaluation times to be within train/test support
        min_time = max(y_train_sksurv["time"].min(), y_test_sksurv["time"].min())
        max_time = min(y_train_sksurv["time"].max(), y_test_sksurv["time"].max())
        
        # If min_time >= max_time, we cannot compute time-dependent metrics
        if min_time >= max_time:
            return metrics
            
        if time_horizons is None:
            # Pick a few percentiles from the test set event times
            mask = y_test_sksurv["event"]
            if sum(mask) > 0:
                time_horizons = np.percentile(y_test_sksurv["time"][mask], [25, 50, 75])
            else:
                time_horizons = []
            
        valid_times = np.array([t for t in time_horizons if min_time < t < max_time])
        
        if len(valid_times) > 0:
            try:
                # 2. Integrated Brier Score
                # Align surv_probs with valid_times via interpolation
                surv_probs_aligned = np.zeros((surv_probs.shape[0], len(valid_times)))
                for i in range(surv_probs.shape[0]):
                    surv_probs_aligned[i, :] = np.interp(valid_times, surv_times, surv_probs[i, :])
                
                ibs = integrated_brier_score(
                    y_train_sksurv, y_test_sksurv, surv_probs_aligned, valid_times
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
                pass
                
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
                    "Mean_Diff": np.mean(diff),
                    "p-value": p_val,
                    "Significant": (p_val < 0.05) if not np.isnan(p_val) else False
                })
                
    return pd.DataFrame(stats_res)
